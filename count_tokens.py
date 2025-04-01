import tiktoken

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("deepseek-chat")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        print(count_tokens(text))
    else:
        print("Please provide a string as a command-line argument.")
