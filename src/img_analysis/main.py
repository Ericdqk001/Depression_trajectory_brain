from scripts.multinomial import run_multinomial
from scripts.prepare_img import prepare_image


def main():
    """Prepare the imaging features and run the multinomial logistic regression."""
    prepare_image()
    run_multinomial()


if __name__ == "__main__":
    main()
