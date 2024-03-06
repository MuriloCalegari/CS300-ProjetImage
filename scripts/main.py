from modules.utils import get_parameter

def main():
    print("""
    Insert option to continue:
          1. Split datasets
          2. Extract labels from dataset
          """)
    
    option = input("Option: ")

    match option:
        case "1":
            from modules.dataset.dataset_manipulation import split_and_persist_datasets
            split_and_persist_datasets(get_parameter("image_path"))
        case "2":
            from modules.dataset.dataset_manipulation import get_labels
            print(get_labels())
        case _:
            print("Invalid option")
            main()

main()