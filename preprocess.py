import pandas as pd
import numpy as np
import time
import NE

def loaddata(data_path):
    start_time = time.time()
    # Original shape:
    data = pd.read_csv(data_path, sep='\t', index_col=0).T  #行细胞，列基因
    #print(type(data))
    genes = data.columns.values
    cells = data.index.values
    data = data.values
    # data = others.normalization(data)
    # After transformation: [n_cell, n_gene]
    print('------------data loaded!  took %f seconds in total------------\n' % (time.time() - start_time))
    return data,genes,cells

def clean(data,genes,cells,lables):
    start_time = time.time()
    print('Before filtering...')
    print(' Number of cells is {}'.format(len(cells)))
    print(' Number of genes is {}'.format(len(genes)))
    #print(data)

    # Filter low-quality cells
    num_Gene = []
    #print('len(data) is {}'.format(len(data)))
    for i in range(len(data)):
        num_Gene.append(len(np.argwhere(data[i] > 0)))
    nGene = np.array(num_Gene)
    #print(num_Gene)

    num_Gene_Q1 = np.percentile(num_Gene, 25)
    num_Gene_Q3 = np.percentile(num_Gene, 75)
    num_Gene_IQR = num_Gene_Q3 - num_Gene_Q1
    num_Gene_high_v = num_Gene_Q3 + 3 * num_Gene_IQR
    num_Gene_low_v = num_Gene_Q1 - 3 * num_Gene_IQR
    # print(high_v)
    # print(low_v)
    # print('threshold is {}~{}'.format(high_v,low_v))
    x = np.argwhere(nGene <= num_Gene_high_v)
    y = np.argwhere(nGene >= num_Gene_low_v)
    save_index1 = np.intersect1d(x, y)
    #print('save index1 is {}'.format(save_index1))
    x1 = np.argwhere(nGene >= num_Gene_high_v)
    y1 = np.argwhere(nGene <= num_Gene_low_v)
    clean_index1 = np.union1d(x1, y1)
    #print('#1 cleaned cell number is {}'.format(len(clean_index1)))

    sum_gene = np.sum(data, axis=1)
    sum_gene_Q1 = np.percentile(sum_gene, 25)
    sum_gene_Q3 = np.percentile(sum_gene, 75)
    sum_gene_IQR = sum_gene_Q3 - sum_gene_Q1
    sum_gene_high_v = sum_gene_Q3 + 3 * sum_gene_IQR
    sum_gene_low_v = sum_gene_Q1 - 3 * sum_gene_IQR
    x = np.argwhere(sum_gene <= sum_gene_high_v)
    y = np.argwhere(sum_gene >= sum_gene_low_v)
    save_index2 = np.intersect1d(x, y)
    # print('save index2 is {}'.format(save_index2))
    x1 = np.argwhere(sum_gene >= sum_gene_high_v)
    y1 = np.argwhere(sum_gene <= sum_gene_low_v)
    clean_index2 = np.union1d(x1, y1)

    clean_index = np.union1d(clean_index1,clean_index2)
    print('#1 cleaned cell number is {}'.format(len(clean_index)))

    index = []
    for i in range(len(save_index1)):
        if save_index1[i] in save_index2:
            index.append(save_index1[i])
    save_index = np.array(index).reshape(-1)
    lables = lables.values.reshape(-1)
    #print(type(save_index1))
    data = data[save_index]
    cells = cells[save_index]
    lables = lables[save_index]
    data = data.T      #行基因，列细胞
    lables = lables.T

    # Filter low-quality genes
    num_Cell = []
    for i in range(len(data)):
        num_Cell.append(len(np.argwhere(data[i] > 0)))
    num_Cell = np.array(num_Cell)
    save_index1 = np.argwhere(num_Cell >= 3)
    clean_index1 = np.argwhere(num_Cell < 3)

    sum_Cell = np.sum(data, axis=1)
    save_index2 = np.argwhere(sum_Cell >= 3)
    clean_index2 = np.argwhere(sum_Cell < 3)

    clean_index = np.union1d(clean_index1, clean_index2)
    print('#2 cleaned gene number is {}'.format(len(clean_index)))

    index = []
    for i in range(len(save_index1)):
        if save_index1[i] in save_index2:
            index.append(save_index1[i])
    save_index = np.array(index).reshape(-1)

    data = data[save_index]
    genes = genes[save_index]
    print('Pre-processing finished')
    print('After filtering...')
    print(' Number of cells is {}'.format(len(cells)))
    print(' Number of genes is {}'.format(len(genes)))

    # Save data
    data = pd.DataFrame(data, index=genes, columns=cells)
    data = data.T   #行细胞，列基因
    print('------------data cleaned!  took %f seconds in total------------\n' % (time.time() - start_time))
    return data,lables


