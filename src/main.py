import argparse
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # TODO: 모델 로딩
    # TODO: 토큰 식별
    # TODO: 토큰 조합 분석
    # TODO: 가중치 조정
    # TODO: 모델 저장

    parser = argparse.ArgumentParser(description="Smoothie-Qwen: Token weight smoothing for language suppression.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loaded configuration:")
    print(config)

    # TODO: 이후 config를 기반으로 전체 파이프라인 진행

if __name__ == "__main__":
    main()