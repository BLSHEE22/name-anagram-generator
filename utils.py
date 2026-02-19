# ===============================================
# --- Graph number of words inside given name ---
# ===============================================

def draw_histogram(data):
    # set chart size
    max_count = max(count for _, count in data)
    height = 20  # number of rows for the tallest bar
    col_width = 6

    # scale counts to fit height
    scaled = [(length, int(count / max_count * height)) for length, count in data]
    print("\nNumber of Words in Name By Word Length Descending\n")

    # print bar labels
    print("".join(str(count).center(col_width) for _, count in data))
    print()

    # build rows top-down
    for level in range(height, 0, -1):
        row = ""
        for _, value in scaled:
            if value >= level:
                row += " █ ".center(col_width)
            else:
                row += "   ".center(col_width)
        print(row)

    # print separator
    print("―" * (len(data) * col_width))

    # print labels
    print("".join(str(length).center(col_width) for length, _ in scaled))
    print("\n       Word Length In Letters       \n\n\n")