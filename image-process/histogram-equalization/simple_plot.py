import sys
import matplotlib.pyplot as plt
import re

bins = []
frequencies = []


if len(sys.argv) != 2:
    print("Usage: python simple_plot.py <filename>")
    sys.exit(1)

filename = sys.argv[1]


with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        # Parse lines like: [000-000]	*********************
        if line.startswith('['):
            # Extract bin number (use start of range)
            match = re.match(r'\[(\d+)-(\d+)\]', line)
            if match:
                bin_value = int(match.group(1))
                # Count asterisks
                asterisks = line.split('\t')[1] if '\t' in line else ''
                frequency = asterisks.count('*')

                bins.append(bin_value)
                frequencies.append(frequency)

plt.figure(figsize=(10, 6))
plt.plot(bins, frequencies, 'b-', linewidth=2)
plt.xlabel('Pixel')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True, alpha=0.3)
plt.xlim(0, 255)
plt.savefig(f'{filename}.png', dpi=300)
plt.show()

print(f"Plotted {len(bins)} data points")
print(
    f"Max frequency: {max(frequencies)} at intensity {bins[frequencies.index(max(frequencies))]}")
