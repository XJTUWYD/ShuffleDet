#! /usr/bin/python3

import numpy as np
import sys
import string
import tensorflow as tf

def next_comb(comb, k, n):
    i = k - 1
    comb[i] += 1

    while i > 0 and comb[i] >= n - k + 1 + i:
        i -= 1
        comb[i] += 1

    if comb[0] > n - k:
        return False

    i += 1
    while i < k:
        comb[i] = comb[i - 1] + 1
        i += 1

    return True

def fact(n, nitem):
    ret = n
    nitem -= 1
    while nitem != 0:
        ret *= n - 1
        nitem -= 1
        n -= 1
    return ret

def powcomb_array(bitwidth, pow_low, pow_high):
    pow_width = pow_high - pow_low + 1
    pow_array = np.zeros((pow_width))
    d1 = fact(pow_width, pow_width - bitwidth) / fact(pow_width - bitwidth, pow_width - bitwidth)
    d2 = bitwidth
    comb = np.zeros((d1, d2))
    index = np.zeros((d2), dtype = np.int)

    for i in range(pow_low, pow_high + 1):
        pow_array[i] = 2 ** i
    for i in range(0, bitwidth):
        index[i] = i
        comb[0][i] = pow_array[index[i]]
    for i in range(1, d1):
        next_comb(index, d2, pow_width)
        for j in range(0, d2):
            comb[i][j] = pow_array[index[j]]

    return comb

def bsearch(array, w):
    # array_sort = np.sort(array)
    high = array.shape[0] - 1
    low = 0
    ret = 0

    while (low <= high):
        high = array.shape[0] - 1
        low = 0
        mid = int(np.floor((high - low) / 2 + low))
        # print high, mid, low, array.shape[0]
        if mid == low or abs(w) == array[mid]:
            if abs(abs(w)-array[mid]) > abs(abs(w)-array[high]):
                ret = array[high]
            else:
                ret = array[mid]
            if w < 0:
                ret = -ret
            return ret

        if abs(w) > array[mid]:
            low = mid
            array = array[mid:high+1]
        elif abs(w) < array[mid]:
            high = mid
            array = array[low:mid]

def parse_tensor_str(tensor_str):
    # print (tensor_str)
    strre = r'(?<=\d)(?=\s)'
    parsed_str = re.sub(strre, ',', tensor_str)
    strre = r'\](\s*)\['
    m = re.search(strre, parsed_str)
    if m:
        parsed_str = re.sub(strre, '],' + m.group(1) + '[', parsed_str)
    # print (parsed_str)
    tensor = eval(parsed_str)
    return tensor

# not work, don't use
def float2fix_online(bitwidth, pow_low, pow_high):
    pow_comb_array = powcomb_array(bitwidth, pow_low, pow_high)
    floats = np.sum(pow_comb_array, axis = 1)
    sorted_array = np.sort(floats.append(0))
    fixed_ops = []
    for x in tf.trainable_variables():
        fix_op = bsearch(sorted_array, x)
        fixed_ops.append(fix_op)

    return tf.group(*fixed_ops)

def float2fix_offline(bitwidth, pow_low, pow_high, data_dir):
    os.system("../../scripts/float2fix.sh -p 0 -b " + bitwidth + " -r " + pow_low + " " + pow_high + " -f " + data_dir + "/*")

def convert_from_file(bitwidth, pow_low, pow_high, filenames):
    pow_comb_array = powcomb_array(bitwidth, pow_low, pow_high)
    # print pow_comb_array
    floats = np.sort(np.sum(pow_comb_array, axis = 1))
    # print floats
    # print "Converting files: ", filenames
    for filename in filenames:
        print ("Converting file \"" + filename + "\"...")
        lines = []
        outputs = []
        try:
            infile = open(filename)
            lines = infile.readlines()
        except IOError:
            print ("No such file \"" + filename + "\"...")
            continue
        finally:
            infile.close()
        for line in lines:
            f = string.atof(line)
            if f == 0:
                outputs.append(0)
                continue;
            outputs.append(bsearch(floats, f))

        outfile = open(filename+"_convert", 'w')
        print (outfile, "\n".join(map(str, outputs)))
        # print "Converted file saved in \"" + filename + "_convert" + "\""


bitwidth = int(sys.argv[1])
pow_low = int(sys.argv[2])
pow_high = int(sys.argv[3])
filenames = sys.argv[4:]
convert_from_file(bitwidth, pow_low, pow_high, filenames)
