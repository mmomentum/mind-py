# this file handles the scope of preprocessing of the CSV
# file itself, data augmentation will be done elsewhere


# returns a string of integers which have been mapped up
# according to the order of appearance within |ref_array|.
# (ex. if |ref_array| was [one, two, three] and  |str_array|
# was [two, three, one], it would return [2 3 1].
def activations2int(ref_array, string):
    string = string.replace(",", "")  # remove commas
    string = string.replace("  ", " ")  # handle double spacing errors
    string = string.replace("   ", " ")  # and triple spacing because why not
    string = string.replace("'", "")

    string = string.replace("dynamics", "dynamic")
    string = string.replace("synthesized", "synthetic")

    # important: remove whitespace
    string = string.strip()

    # convert sentence to string array
    start = 0
    i = 0
    return_array = []
    for i in range(len(string)):
        if " " == string[i:i + 1]:
            return_array.append(string[start:i + 1])
            start = i + 1
        # i += 1
    return_array.append(string[start:i + 1])

    d = dict([(y, x) for x, y in enumerate(ref_array)])

    # need to remove spaces for conversion
    for i in range(len(return_array)):
        return_array[i] = return_array[i].strip()

    return_array = [d[x] for x in return_array]

    # join the list of integers to a single string
    return ' '.join([str(x) for x in return_array])


# adds a suffix to the given filename if it is not
# already there; used for adding .wav suffix in this case.
def add_suffix(filename, suffix):
    if filename[-len(suffix):] != suffix:
        return filename + suffix
    else:
        return filename
