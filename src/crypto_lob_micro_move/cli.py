from crypto_lob_micro_move.data import download
from crypto_lob_micro_move import train as train_module

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)

    def train(self, data_dir: str = "data/raw", epochs: int = 1):
        """Run a stub training routine."""
        train_module.train(data_dir=data_dir, epochs=epochs)


def main():
    import fire

    fire.Fire(Commands)

if __name__ == "__main__":
    main()
