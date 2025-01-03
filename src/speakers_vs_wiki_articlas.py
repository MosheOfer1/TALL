import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data_sets/speakers_vs_articles.csv'
data = pd.read_csv(file_path)

# Filter the dataset for selected languages
selected_languages = [
    'Chinese', 'English', 'Arabic', 'Bengali', 'Hindi',
    'Telugu', 'Swahili', 'Hebrew'
]
filtered_data = data[data['Language name(in English)'].isin(selected_languages)].copy()

# Convert 'Speakers(L1 & L2)' and 'Articles' to numeric values
filtered_data.loc[:, 'Speakers(L1 & L2)'] = pd.to_numeric(
    filtered_data['Speakers(L1 & L2)'].str.replace(',', ''), errors='coerce'
)
filtered_data.loc[:, 'Articles'] = pd.to_numeric(
    filtered_data['Articles'].str.replace(',', ''), errors='coerce'
)

# Convert values to millions
filtered_data.loc[:, 'Speakers (Millions)'] = filtered_data['Speakers(L1 & L2)'] / 1e6
filtered_data.loc[:, 'Articles (Millions)'] = filtered_data['Articles'] / 1e6

# Sort the data by number of speakers in descending order
filtered_data = filtered_data.sort_values('Speakers (Millions)', ascending=False)

# Plotting
plt.style.use('classic')
fig, ax1 = plt.subplots(figsize=(18, 9))

# Set background color to light gray for better contrast
fig.patch.set_facecolor('#f0f0f0')
ax1.set_facecolor('#f0f0f0')

# Bar widths
bar_width = 0.4

# X positions
x_positions = range(len(filtered_data))

# Plot Speakers as bars
ax1.bar(x_positions, filtered_data['Speakers (Millions)'],
        color='#4477AA', alpha=0.7, label='Speakers (Millions)', width=bar_width)
ax1.set_xlabel('', fontsize=18, labelpad=15)
ax1.set_ylabel('First & Second Language\nSpeakers (Millions)', color='#4477AA', fontsize=22, labelpad=15)
ax1.tick_params(axis='y', labelcolor='#4477AA', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(filtered_data['Language name(in English)'], fontsize=22, rotation=0, ha='left')

# Plot Articles as bars
ax2 = ax1.twinx()
ax2.bar([x + bar_width for x in x_positions], filtered_data['Articles (Millions)'],
        color='#EE6677', alpha=0.7, label='Articles (Millions)', width=bar_width)
ax2.set_ylabel('Articles (Millions)', color='#EE6677', fontsize=24, labelpad=15)
ax2.tick_params(axis='y', labelcolor='#EE6677', labelsize=22)

# Add grid for better readability
ax1.grid(True, axis='y', alpha=0.3)

# Add title and source
plt.title('Speakers vs Wikipedia Articles\nby Language',
          fontsize=30, fontweight='bold', pad=40)
plt.figtext(0.5, 0.02,
            'Data represents the total number of first and second language speakers and Wikipedia articles.\nSource: meta.wikimedia.org/wiki/List_of_Wikipedias_by_speakers_per_article',
            wrap=True, horizontalalignment='center', fontsize=24, color='#666666',
            fontstyle='italic')

# Add a legend with larger font
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
          fontsize=18, frameon=True, bbox_to_anchor=(1, 1))

# Improve layout
fig.tight_layout(rect=[0, 0.07, 1, 0.95])

# Save with higher resolution
plt.savefig('speakers_vs_articles.png', dpi=500, bbox_inches='tight')
plt.show()