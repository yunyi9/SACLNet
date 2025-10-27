import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family='Times New Roman')
categories = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']

SARA = [65.22, 67.77, 65.69, 64.94, 62.62, 63.59]
Baseline = [52.17, 47.11, 53.28 ,45.45, 55.14, 57.61]

STEANet = np.concatenate((SARA, [SARA[0]]))
Baseline = np.concatenate((Baseline, [Baseline[0]]))

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
print(angles)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

ax.plot(angles, Baseline, 'o-', linewidth=2, label='FTCH-UniFormerV2', color='#111111')
ax.plot(angles, STEANet, 'o-', linewidth=2, label='SACLNet', color='#D04846')

ax.fill(angles, Baseline, alpha=0.2, color='#111111')
ax.fill(angles, STEANet, alpha=0.2, color='#BF0753')
for i in range(len(categories)):
    ax.text(angles[i], Baseline[i] - 8, f'{Baseline[i]:.2f}',
            ha='center', va='center', fontsize=24, color='#111111')
    ax.text(angles[i], STEANet[i] + 4, f'{STEANet[i]:.2f}',
            ha='center', va='center', fontsize=24, color='#BF0753')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=24)

for i, angle in enumerate(angles[:-1]):
    ax.set_rgrids([40, 48, 56, 64, 72], labels=[], angle=angle, color="black", alpha=0.7)

ax.set_ylim(0, 79)
ax.tick_params(pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=24)

plt.tight_layout()
plt.savefig('radargram_EK6.jpg', dpi=300)
plt.show()
