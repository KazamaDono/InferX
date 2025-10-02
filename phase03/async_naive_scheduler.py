import asyncio
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class AsyncNaiveScheduler:
    """
    Asynchronous batch scheduler for LLM inference.
    Collects multiple requests and processes them in batches for efficiency.
    """

    def __init__(
        self, modelname, batch_size: int = 4, device="cpu", timeout: float = 0.5
    ):
        """
        Initialize the scheduler.

        Args:
            modelname: HuggingFace model identifier
            batch_size: Maximum number of requests per batch
            device: Device to run inference on ('cpu' or 'cuda')
            timeout: Maximum seconds to wait for batch to fill
        """
        self.modelname = modelname
        self.batch_size = batch_size
        self.device = device
        self.timeout = timeout

        # Queue to hold incoming (prompt, future) pairs
        self.queue = asyncio.Queue()
        self.running = True
        self.batch_count = 0

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # FIX: Set pad token
        self.model.to(self.device)  # FIX: Move model to device

        # Start the background batch processing loop
        self.task = asyncio.create_task(self._batch_loop())

    async def add_request(self, prompt: str):
        """
        Add a request to the processing queue.

        Args:
            prompt: Input text prompt

        Returns:
            Generated text (awaits until batch is processed)
        """
        # Create a future to wait for the result
        fut = asyncio.get_event_loop().create_future()

        # Add to queue
        await self.queue.put((prompt, fut))

        # Wait for result (blocks until future is resolved by _batch_loop)
        return await fut

    async def _batch_loop(self):
        """
        Background loop that continuously processes batches.
        Collects requests until batch_size is reached or timeout expires.
        """
        while self.running:
            batch = []
            try:
                # Record start time for this batch
                start_time = asyncio.get_event_loop().time()

                # Collect items until batch is full or timeout
                while len(batch) < self.batch_size:
                    # FIX: Calculate remaining time correctly
                    timeout_remaining = self.timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )

                    # If timeout expired and we have items, process what we have
                    if timeout_remaining <= 0 and batch:
                        break

                    try:
                        # Wait for next item with remaining timeout
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=max(0.01, timeout_remaining)
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        # Timeout reached
                        if batch:
                            # We have some items, process them
                            break
                        # No items yet, continue waiting
                        continue

                # Skip if no items collected
                if not batch:
                    continue

                # Separate prompts and futures
                prompts = [prompt for prompt, _ in batch]
                futures = [fut for _, fut in batch]

                # Run inference on the batch
                decoded = self.run_batch(prompts)

                # Set results for all futures
                for fut, out in zip(futures, decoded):
                    if not fut.done():
                        fut.set_result(out)

                self.batch_count += 1

            except Exception as e:
                # Handle errors by setting exception on all futures in batch
                print(f"Error in batch loop: {e}")
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def run_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts and return generated text.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated text (one per prompt)
        """
        # Tokenize all prompts with padding
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )

        # Get input length to extract only new tokens later
        input_length = inputs["input_ids"].shape[1]

        # Generate text
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        # Only decode the newly generated tokens (exclude input prompt)
        new_tokens = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return decoded

    async def shutdown(self):
        """
        Gracefully shutdown the scheduler.
        Processes any remaining items in queue before stopping.
        """
        # Signal the loop to stop
        self.running = False

        # Collect remaining items from queue
        remaining_batch = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                remaining_batch.append(item)
            except asyncio.QueueEmpty:
                break

        # Process remaining items if any
        if remaining_batch:
            try:
                prompts = [prompt for prompt, _ in remaining_batch]
                futures = [
                    fut for _, fut in remaining_batch
                ]  # FIX: Better variable name
                decoded = self.run_batch(prompts)

                # Set results for remaining futures
                for fut, out in zip(futures, decoded):  # FIX: Use 'futures' not 'fut'
                    if not fut.done():
                        fut.set_result(out)
            except Exception as e:
                # FIX: Set exception on all remaining futures
                for _, fut in remaining_batch:
                    if not fut.done():
                        fut.set_exception(e)

        # Cancel and await the background task
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
