from src import Translation


def example():
    service = Translation(load_type='local', model_path='models/Qwen2.5-3B-Instruct')
    result = service.translate("Hello, how are you?")
    print(f"Translation: {result}")
    print(f"Time: {service.last_translation_time:.3f}s")


    result = service.translate("你好世界", tgt_lang="eng_Latn")
    print(f"Translation: {result}")
    print(f"Time: {service.last_translation_time:.3f}s")


    texts = ["Hello", "Goodbye", "Thank you"]
    results = service.translate(texts, json_output_path='data/output.json')
    for src, tgt in zip(texts, results):
        print(f"{src} -> {tgt}")
    print(f"Total Time: {service.last_batch_time:.3f}s")


def example2():
    service = Translation(load_type='local', model_path='models/Qwen2.5-3B-Instruct', src_lang='zho_Hans', tgt_lang="eng_Latn")
    texts = ["请减速至40km/h", "前方路口处刹车", "先左转再加速至70km/h"]
    results = service.translate(texts, json_output_path='data/output.json')
    for src, tgt in zip(texts, results):
        print(f"{src} -> {tgt}")
    print(f"Total Time: {service.last_batch_time:.3f}s")


if __name__ == "__main__":
    example2()
