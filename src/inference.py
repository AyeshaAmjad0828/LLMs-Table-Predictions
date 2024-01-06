import modal
import time

from .common import stub, vllm_image, VOLUME_CONFIG


@stub.cls(
    gpu="A100",
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=120,
)
class Inference:
    def __init__(self, model_path: str) -> None:
        print("Initializing vLLM engine on:", model_path)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(model=model_path, gpu_memory_utilization=0.95)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def completion(self, input: str):
        if not input:
            return

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)


@stub.local_entrypoint()
def inference_main(run_folder: str):
    text = input(
        "Below is an instruction that describes a task. Write an astera expression in response to complete the request.\n"
    )
    print("Loading model ...")
    for chunk in Inference(f"{run_folder}/lora-out/merged").completion.remote_gen(text):
        print(chunk, end="")