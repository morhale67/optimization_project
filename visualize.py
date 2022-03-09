import matplotlib.pyplot as plt

def style_image(x_label, y_label, fs1=16, fs2=20):
  plt.rcParams["font.family"] = "serif"
  plt.xticks(fontsize=fs1)
  plt.yticks(fontsize=fs1)
  plt.xlabel(x_label, fontsize=fs2, fontweight='bold')
  plt.ylabel(y_label, fontsize=fs2, fontweight='bold')