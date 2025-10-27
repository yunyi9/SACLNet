import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family='Times New Roman')
categories = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness',
              'Surprise', 'Trust']
STEANet = [69.7, 27.27, 53.85, 72.22, 58.33, 66.67, 74.07, 37.5]
Baseline = [33.33, 6.06, 25.64, 48.15, 30, 15.15, 74.07, 18.75]

Baseline = np.concatenate((Baseline, [Baseline[0]]))
STEANet = np.concatenate((STEANet, [STEANet[0]]))

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
print(angles)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, Baseline, 'o-', linewidth=2, label='FTCH-UniFormerV2', color='#111111')
ax.plot(angles, STEANet, 'o-', linewidth=2, label='SACLNet', color='#D04846')
ax.fill(angles, Baseline, alpha=0.2, color='#111111')
ax.fill(angles, STEANet, alpha=0.2, color='#BF0753')
for i in range(len(categories)):
    ax.text(angles[i], Baseline[i] + 4, f'{Baseline[i]:.2f}',
            ha='center', va='center', fontsize=24, color='#111111')
    ax.text(angles[i], STEANet[i] + 4, f'{STEANet[i]:.2f}',
            ha='center', va='center', fontsize=24, color='#BF0753')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=24)

for i, angle in enumerate(angles[:-1]):
    ax.set_rgrids([25, 40, 55, 70, 85], labels=[], angle=angle, color="black", alpha=0.7)
ax.set_ylim(0, 87.5)
ax.tick_params(pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=24)

plt.tight_layout()
plt.savefig('radargram_VE8.jpg', dpi=300)
plt.show()