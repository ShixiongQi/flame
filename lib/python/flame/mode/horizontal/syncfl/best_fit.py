import time

num_bins = 50

def best_fit(bin_capacity, items):
    bins = [[] for _ in range(num_bins)]  # Initialize 5 bins

    for item in items:
        best_bin_index = None
        min_waste = float('inf')

        for i in range(num_bins):
            if sum(bins[i]) + item <= bin_capacity:
                waste = bin_capacity - (sum(bins[i]) + item)
                if waste < min_waste:
                    best_bin_index = i
                    min_waste = waste

        if best_bin_index is not None:
            bins[best_bin_index].append(item)
        else:
            # If no bin can accommodate the item, create a new bin
            new_bin = [item]
            bins.append(new_bin)

    return bins

# Example usage
bin_capacity = 20
num_items = 1000
items = [1] * num_items

START_T = time.time()
packed_bins = best_fit(bin_capacity, items)
END_T = time.time()

print(f"Placement delay: {END_T - START_T:.5f}")

for i, bin in enumerate(packed_bins):
    print(f'Bin {i + 1}: {bin}')

print(f'Total bins used: {len(packed_bins)}')