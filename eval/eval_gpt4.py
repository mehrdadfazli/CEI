import os
import argparse
import json
import time

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    parser.add_argument(
        '--summary_file',
        type=str,
        default=None,
        help='Optional JSON path: average score, hallucination rate, per-question-type means.',
    )
    parser.add_argument('--gpt-model', type=str, default='gpt-4o')
    parser.add_argument(
        '--api-provider',
        type=str,
        choices=['openai', 'openrouter', 'custom'],
        default='openai',
        help='LLM API provider. Use openrouter for OpenRouter-hosted GPT-4o.',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key override. Falls back to OPENAI_API_KEY, then OPENROUTER_API_KEY.',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Base URL override (e.g. https://openrouter.ai/api/v1).',
    )
    parser.add_argument(
        '--openrouter-http-referer',
        type=str,
        default=None,
        help='Optional OpenRouter HTTP-Referer header (or env OPENROUTER_HTTP_REFERER).',
    )
    parser.add_argument(
        '--openrouter-x-title',
        type=str,
        default=None,
        help='Optional OpenRouter X-Title header (or env OPENROUTER_X_TITLE).',
    )
    parser.add_argument(
        '--request-sleep',
        type=float,
        default=1.0,
        help='Seconds to sleep between judge requests.',
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("API key required. Set --api-key or OPENAI_API_KEY/OPENROUTER_API_KEY.")

    base_url = args.base_url
    default_headers = None
    if args.api_provider == "openrouter":
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        referer = args.openrouter_http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        title = args.openrouter_x_title or os.environ.get("OPENROUTER_X_TITLE")
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        default_headers = headers or None
    elif args.api_provider == "openai":
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    else:
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
        headers = {}
        if args.openrouter_http_referer:
            headers["HTTP-Referer"] = args.openrouter_http_referer
        if args.openrouter_x_title:
            headers["X-Title"] = args.openrouter_x_title
        default_headers = headers or None

    # Build a version-agnostic chat function (supports openai v0.x and v1.x)
    try:
        from openai import OpenAI  # v1.x
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        _client = OpenAI(**client_kwargs)

        def _chat(prompt_text):
            resp = _client.chat.completions.create(
                model=args.gpt_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()

    except ImportError:
        import openai as _oa  # v0.x
        _oa.api_key = api_key
        if base_url:
            _oa.api_base = base_url

        def _chat(prompt_text):
            resp = _oa.ChatCompletion.create(
                model=args.gpt_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()

    # load json file
    with open(args.response, 'r') as f:
        records = json.load(f)

    assert len(records) == 96

    # ask GPT-4 to evaluate
    responses = []
    for i, record in enumerate(records):
        image_content = ', '.join(record['image_content'])
        input_text = template.format(image_content, record['question'], record['gt_answer'], record['model_answer'])

        response_text = None
        while response_text is None:
            try:
                response_text = _chat(input_text)
            except Exception as e:
                print(e)
                print('retrying...')
                time.sleep(10)
                continue

        print(i, response_text, flush=True)
        responses.append({"content": response_text, "model": args.gpt_model})
        time.sleep(args.request_sleep)

    # save responses
    if args.evaluation is not None:
        with open(args.evaluation, 'w') as f:
            json.dump(responses, f, indent=2)

    # analyze responses
    scores = []
    for i, response in enumerate(responses):
        if isinstance(response, dict):
            if "content" in response:
                response = response["content"]
            elif "choices" in response:
                response = response["choices"][0]["message"]["content"]
            else:
                response = str(response)
        else:
            response = str(response)
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    avg_score = sum(scores) / len(scores)
    hallu_rate = sum(hallucination) / len(hallucination)
    per_type = [round(sum(scores_each[i]) / len(scores_each[i]), 4) for i in range(8)]

    print('Average score: {:.2f}'.format(avg_score))
    print('Hallucination rate: {:.2f}'.format(hallu_rate))
    print('Average score for each question type:', ','.join([str(x) for x in per_type]), flush=True)

    if args.summary_file:
        parent = os.path.dirname(args.summary_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        summary = {
            "response_file": args.response,
            "gpt_model": args.gpt_model,
            "api_provider": args.api_provider,
            "num_samples": len(scores),
            "average_score": round(avg_score, 4),
            "hallucination_rate": round(hallu_rate, 4),
            "average_score_by_question_type_index": per_type,
        }
        with open(args.summary_file, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
