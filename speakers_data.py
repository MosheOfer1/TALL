import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

# Set the style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Load and parse the data
file_path = 'table class wikitable.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

soup = BeautifulSoup(content, 'html.parser')
headers = [header.text.strip() for header in soup.find_all('th')]
rows = soup.find_all('tr')[1:]
data = []
for row in rows:
    columns = row.find_all(['td', 'th'])
    data.append([col.text.strip() for col in columns])

# Create and clean DataFrame
df = pd.DataFrame(data, columns=headers)
df['Speakers(L1 & L2)'] = pd.to_numeric(df['Speakers(L1 & L2)'].str.replace(',', ''), errors='coerce')
df['Articles'] = pd.to_numeric(df['Articles'].str.replace(',', ''), errors='coerce')

# Select interesting languages
selected_languages = [
    'Chinese',
    'English',
    'Arabic',
    'Bengali',
    'Hindi',
    'Telugu',
    'Swahili',
    'Hebrew',
    'Estonian',
    'Icelandic',
    'Irish'
]

# Filter and prepare data
plot_df = df[df['Language name(in English)'].isin(selected_languages)].copy()
plot_df = plot_df.sort_values('Speakers(L1 & L2)', ascending=True)

# Create figure
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

# Set the width of bars and positions of the bars
bar_width = 0.35
y_pos = np.arange(len(plot_df))

# Create bars
speakers_bars = ax.barh(y_pos - bar_width/2, plot_df['Speakers(L1 & L2)'],
                       bar_width, label='Number of Speakers',
                       color='#2C3E50', alpha=0.8)

articles_bars = ax.barh(y_pos + bar_width/2, plot_df['Articles'],
                       bar_width, label='Number of Articles',
                       color='#E74C3C', alpha=0.8)

# Format numbers for labels
def format_number(x):
    if x >= 1e9:
        return f'{x/1e9:.1f}B'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.1f}K'
    return f'{x:.0f}'

# Add value labels on bars
for bars in [speakers_bars, articles_bars]:
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                format_number(width),
                va='center', fontsize=10,
                color='#34495E',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# Customize axes
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['Language name(in English)'], fontsize=11)

# Format x-axis
def format_axis(x, p):
    return format_number(x)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis))

# Add title and labels
ax.set_title('Wikipedia Language Resources: Articles vs. Speaker Population',
             fontsize=14, fontweight='bold', pad=20)

# Add legend
ax.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=2,
         fontsize=11, frameon=True)

# Customize grid
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Add explanatory note
plt.figtext(0.01, 0.02,
            'Note: This visualization compares Wikipedia article count with speaker population for selected languages,\n' +
            'highlighting the disparity between language resources and speaker base.',
            fontsize=8, ha='left')

# Add source citation
plt.figtext(0.01, -0.02,
            'Source: Wikimedia Foundation (2024). List of Wikipedias by speakers per article.\n' +
            'Retrieved from meta.wikimedia.org/wiki/List_of_Wikipedias_by_speakers_per_article',
            fontsize=7, ha='left')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15)

# Save the figure
plt.savefig('wikipedia_language_comparison_parallel.png', dpi=300, bbox_inches='tight')
plt.show()