import sys
from termios import tcflush, TCIFLUSH
from llmtuner import ChatModel


def main():
    chat_model = ChatModel()
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            tcflush(sys.stdin, TCIFLUSH)
            print("-" * 20 + " Use Ctrl-C to Exit " + "-" * 20)

            # get the instruction
            print("\n> Instruction: ( Use Ctrl-D to Stop Input)")
            instruction = ""
            try:
                while True:
                    instruction_line = input() + "\n"
                    instruction += instruction_line
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                continue
            except EOFError:
                pass
            if not instruction.strip():
                continue
        except Exception:
            raise

        print("Assistant: \n", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(instruction):
            print(new_text, end="", flush=True)
            response += new_text
        print()


if __name__ == "__main__":
    main()
