import fire
from crypto_lob_micro_move.data import download

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)


def main():
    fire.Fire(Commands)

if __name__ == "__main__":
    main()
