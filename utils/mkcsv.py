import pandas as pd

# 加载现有的数据
data = {
    'Simulation': [
        'No Processing', 'No Processing', 'No Processing', 'No Processing',
        'Rotate 5 degrees', 'Rotate 5 degrees', 'Rotate 5 degrees', 'Rotate 5 degrees',
        'Rotate 15 degrees', 'Rotate 15 degrees', 'Rotate 15 degrees', 'Rotate 15 degrees',
        'Scale +10%', 'Scale +10%', 'Scale +10%', 'Scale +10%',
        'Scale +20%', 'Scale +20%', 'Scale +20%', 'Scale +20%',
        'Noise +15%', 'Noise +15%', 'Noise +15%', 'Noise +15%',
        'Noise +25%', 'Noise +25%', 'Noise +25%', 'Noise +25%'
    ],
    'Image': [
        'blending+blur_1.png', 'blending_1.png', 'p2p+blur_1.png', 'p2p_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png',
        'p2p_1.png', 'p2p_blending_1.png', 'p2p_blending_blur_1.png', 'p2p_blur_1.png'
    ],
    'MSE': [
        14.9244640837476, 14.775065099029433, 14.842370919803257, 14.773760488176965,
        70.75004347022833, 69.92589141136963, 69.79043817990154, 70.73329384787594,
        70.277585, 68.967205, 68.69305, 70.2351,
        78.84064, 74.53155, 74.467895, 78.84064,
        83.58295555555556, 57.26655, 57.827394444444444, 83.84028333333333,
        72.42395118230358, 52.52874615323917, 51.67512822536101, 71.9899471316973,
        80.03480102054236, 61.93146058549672, 61.20544201583419, 79.72731791998737
    ],
    'SSIM': [
        0.9096119106374424, 0.9106700613234181, 0.9103152409091649, 0.9107258583724852,
        0.41564293100858807, 0.4333442989667428, 0.4380434836707853, 0.4181869515311121,
        0.4705027896886632, 0.48906265505892355, 0.4957558012988098, 0.4735403323257863,
        0.38026894750588097, 0.42427766453698756, 0.4262845217813582, 0.38026894750588097,
        0.3726133559568946, 0.7044505098019208, 0.7040862122334809, 0.3723550033136184,
        0.3288335797559233, 0.5614184228271079, 0.572189063102005, 0.33418947013369804,
        0.2792663861901624, 0.488810833542142, 0.4993907669200705, 0.28429924000435397
    ],
    'Pearson': [
        0.9888747581954197, 0.9893085329979063, 0.9890500781105516, 0.9893005794495071,
        0.6557794007448666, 0.6823963938416087, 0.6842002387803687, 0.6568245142225695,
        0.5816945488107683, 0.6193636141478618, 0.6219728204062838, 0.5831584509253087,
        0.4590894102607848, 0.6013545395592875, 0.6022827077141673, 0.4590894102607848,
        0.39237881178003275, 0.7420370340188599, 0.7433136599887011, 0.39191271809716344,
        0.7985305368755917, 0.8862758021831492, 0.8897732804109475, 0.7998664062646095,
        0.6536165910418001, 0.789427938120733, 0.7954920271664807, 0.6554040105341561
    ]
}

df = pd.DataFrame(data)

# 计算相对于 p2p_1.png 的差值
df_baseline = df[df['Image'] == 'p2p_1.png']
df_diff = df.copy()

for simulation in df['Simulation'].unique():
    baseline_values = df_baseline[df_baseline['Simulation'] == simulation].iloc[0]
    df_diff.loc[df_diff['Simulation'] == simulation, 'MSE'] -= baseline_values['MSE']
    df_diff.loc[df_diff['Simulation'] == simulation, 'SSIM'] -= baseline_values['SSIM']
    df_diff.loc[df_diff['Simulation'] == simulation, 'Pearson'] -= baseline_values['Pearson']

# 保存为 CSV 文件
output_csv_path = 'similarity_comparison_diff.csv'
df_diff.to_csv(output_csv_path, index=False)

print(f"Data saved to {output_csv_path}")
