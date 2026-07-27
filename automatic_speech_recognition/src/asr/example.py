from src import FunASRService


def example():
    service = FunASRService(mode='single', device='cuda:0')
    result = service.transcribe('data/wav_files/example.wav', output_json='data/output_single.json')

    service = FunASRService(mode='speaker')
    result = service.transcribe('data/wav_files/example.wav', output_json='data/output_speaker.json', print_result=True)


if __name__ == '__main__':
    example()
