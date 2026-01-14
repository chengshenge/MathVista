import argparse
import io
import logging
import os
import sys
import time

from datasets import load_dataset
from openai import OpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.build_query import create_query_data
from utilities import read_json, save_json


# =========================
# Utils
# =========================
def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    sys.stdout = old_stdout
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    return captured_output, error


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')

    # output
    parser.add_argument('--output_dir', type=str, default='results/gpt4o')
    parser.add_argument('--output_file', type=str, default='output_gpt4o_testmini.json')
    parser.add_argument('--max_num_problems', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=20)

    # model
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-4.1'],
    )
    parser.add_argument('--key', type=str, default='', help='OpenAI API key')

    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--use_ocr', action='store_true')

    # retry / rate limit
    parser.add_argument('--max_retries', type=int, default=10)
    parser.add_argument('--retry_sleep', type=int, default=20,
                        help='seconds to sleep when hitting rate limit')

    # other
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()

    # load dataset
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}")
    data_list = load_dataset(args.dataset_name, split=args.test_split_name)
    data = {item['pid']: item for item in data_list}

    # load / build query
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        logging.info(f"Loading query file {query_file}")
        query_data = read_json(query_file)
    else:
        logging.info("Creating query data")
        caption_data = {}
        ocr_data = {}

        if args.use_caption and os.path.exists(args.caption_file):
            caption_data = read_json(args.caption_file)["texts"]

        if args.use_ocr and os.path.exists(args.ocr_file):
            ocr_data = read_json(args.ocr_file)["texts"]

        query_data = create_query_data(data, caption_data, ocr_data, args)

    # OpenAI client
    api_key = args.key if args.key else os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY not set."

    client = OpenAI(api_key=api_key)

    from models import gpt
    model = gpt.GPT_Model(client=client, model=args.model)

    logging.info(f"Model loaded: {args.model}")

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load existing results
    if os.path.exists(output_file_path):
        logging.info(f"Loading existing results from {output_file_path}")
        results = read_json(output_file_path)
    else:
        results = {}

    # skip finished
    skip_pids = []
    if not args.rerun:
        for pid, res in results.items():
            if 'response' in res and verify_response(res['response']):
                skip_pids.append(pid)

    test_pids = [pid for pid in data if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[:args.max_num_problems]

    logging.info(f"Running {len(test_pids)} problems")

    # =========================
    # Main loop with retry
    # =========================
    for i, pid in enumerate(tqdm(test_pids)):
        problem = data[pid].copy()
        decoded_image = problem.pop('decoded_image')
        query = query_data[pid]

        attempt = 0
        while True:
            try:
                response = model.get_response(
                    user_prompt=query,
                    decoded_image=decoded_image
                )

                results[pid] = problem
                results[pid]['query'] = query
                results[pid]['response'] = response
                break

            except Exception as e:
                msg = str(e)
                attempt += 1

                # rate limit handling
                if 'rate limit' in msg.lower() or '429' in msg:
                    if attempt > args.max_retries:
                        logging.error(f"[{pid}] Exceeded max retries due to rate limit.")
                        results[pid] = problem
                        results[pid]['error'] = msg
                        break

                    logging.warning(
                        f"[{pid}] Rate limit hit. "
                        f"Retry {attempt}/{args.max_retries} after {args.retry_sleep}s"
                    )
                    time.sleep(args.retry_sleep)
                    continue

                # other errors
                logging.error(f"[{pid}] Error: {msg}")
                results[pid] = problem
                results[pid]['error'] = msg
                break

        # periodic save
        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)
            logging.info(f"Saved results to {output_file_path}")

    logging.info("MathVista: Generating Responses - Finish")


# =========================
# Entry
# =========================
if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    for module in [
        "asyncio", "datasets", "httpx", "openai", "PIL", "urllib3"
    ]:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
