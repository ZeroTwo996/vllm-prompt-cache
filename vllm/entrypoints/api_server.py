"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION

import time
import numpy as np
from gptcache import Cache
from gptcache.utils.time import time_cal
from gptcache.utils.log import gptcache_log
from gptcache.utils.error import NotInitError
from gptcache.embedding.bge import BGE
from gptcache.processor.pre import get_prompt
from gptcache.processor.post import temperature_softmax
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.langchain_models import _cache_data_convert, _update_cache_callback

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
llm_cache = None
embed_model = None
_input_summarizer = None


def _summarize_input(text, text_length):
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from gptcache.processor.context.summarization_context import (
        SummarizationContextProcess,
    )

    global _input_summarizer
    if _input_summarizer is None:
        _input_summarizer = SummarizationContextProcess()
    summarization = _input_summarizer.summarize_to_sentence([text], text_length)
    return summarization


def cache_health_check(vectordb, cache_dict):
    """This function checks if the embedding
    from vector store matches one in cache store.
    If cache store and vector store are out of
    sync with each other, cache retrieval can
    be incorrect.
    If this happens, force the similary score
    to the lowerest possible value.
    """
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    emb_in_vec = vectordb.get_embeddings(data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        gptcache_log.critical("Cache Store and Vector Store are out of sync!!!")
        # 0: identical, inf: different
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        # self-healing by replacing entry
        # in the vec store with the one
        # from cache store by the same
        # entry_id.
        vectordb.update_embeddings(
            data_id,
            emb=cache_dict["embedding"],
        )
    return flag


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


@app.post("/generateWithPromptCache")
async def generate_with_prompt_cache(request: Request) -> Response:
    request_dict = await request.json()
    return await _generate_with_prompt_cache(request_dict, raw_request=request)


@with_cancellation
async def _generate_with_prompt_cache(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    
    request_dict["prompt"] = prompt
    request_dict["session"] = None
    request_dict["stop"] = None
    request_id = random_uuid()

    args = {}
    start_time = time.time()
    user_temperature = "temperature" in request_dict
    user_top_k = "top_k" in request_dict
    temperature = request_dict.pop("temperature", 0.0)
    session = request_dict.pop("session", None)
    require_object_store = request_dict.pop("require_object_store", False)
    if require_object_store:
        assert llm_cache.data_manager.o, "Object store is required for adapter."
    if not llm_cache.has_init:
        raise NotInitError()
    cache_enable = llm_cache.cache_enable_func(*args, **request_dict)
    context = request_dict.pop("cache_context", {})
    partition_key = request_dict.pop("partition_key", None)

    # 寻找缓存中相似的Prompt并返回结果
    embedding_data = None
    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        cache_skip = request_dict.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = request_dict.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = request_dict.pop("cache_skip", False)
    cache_factor = request_dict.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        llm_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=llm_cache.report.pre,
    )(
        request_dict,
        extra_param=context.get("pre_embedding_func", None),
        prompts=llm_cache.config.prompts,
        cache_config=llm_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if llm_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, llm_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(
            llm_cache.embedding_func,
            func_name="embedding",
            report_func=llm_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            llm_cache.data_manager.search,
            func_name="search",
            report_func=llm_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=request_dict.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else request_dict.pop("top_k", -1),
            partition_key=partition_key,
        )
        if search_data_list is None:
            search_data_list = []
        cache_answers = []
        similarity_threshold = llm_cache.config.similarity_threshold
        min_rank, max_rank = llm_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        for search_data in search_data_list:
            cache_data = time_cal(
                llm_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=llm_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            # cache consistency check
            if llm_cache.config.data_check:
                is_healthy = cache_health_check(
                    llm_cache.data_manager.v,
                    {
                        "embedding": cache_data.embedding_data,
                        "search_result": search_data,
                    },
                )
                if not is_healthy:
                    continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                llm_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=llm_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                llm_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            hit_callback = request_dict.pop("hit_callback", None)
            if hit_callback and callable(hit_callback):
                factor = max_rank - min_rank
                hit_callback([(d[3].question, d[0] / factor if factor else d[0]) for d in cache_answers])
            def post_process():
                if llm_cache.post_process_messages_func is temperature_softmax:
                    return_message = llm_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = llm_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=llm_cache.report.post,
            )()
            llm_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                llm_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data and not llm_cache.config.disable_report:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3]
                report_search_data = cache_whole_data[2]
                llm_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return _cache_data_convert(return_message) # 找到相似Prompt，直接返回结果

    # 没有找到相似Prompt，继续推理
    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    assert len(final_output.outputs) > 0
    response_text = final_output.outputs[0].text

    # 缓存推理结果
    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    llm_cache.data_manager.save,
                    func_name="save",
                    report_func=llm_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                    partition_key=partition_key,
                )
                if (
                    llm_cache.report.op_save.count > 0
                    and llm_cache.report.op_save.count % llm_cache.config.auto_flush
                    == 0
                ):
                    llm_cache.flush()

            response_text = _update_cache_callback(
                response_text, update_cache_func, *args, **request_dict
            )
        except Exception as e:  # pylint: disable=W0703
            gptcache_log.warning("failed to save the data to cache, error: %s", e)

    return JSONResponse({"text": response_text})

def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


def init_prompt_cache():
    global llm_cache, embed_model

    embed_model = (embed_model if embed_model is not None else BGE())
    llm_cache = (llm_cache if llm_cache is not None else Cache())
    llm_cache.init(
        embedding_func=embed_model.to_embeddings,
        data_manager=get_data_manager(
            CacheBase("sqlite"), 
            VectorBase("faiss", dimension=embed_model.dimension), 
            max_size=100000),
        pre_embedding_func=get_prompt,
        post_process_messages_func=temperature_softmax,
        similarity_evaluation=SearchDistanceEvaluation(),
    )


async def init_app(
    args: Namespace,
    llm_engine: Optional[AsyncLLMEngine] = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER))
    init_prompt_cache()
    return app


async def run_server(args: Namespace,
                     llm_engine: Optional[AsyncLLMEngine] = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
