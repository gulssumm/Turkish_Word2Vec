import pandas as pd

df = pd.read_excel("TR_Commands.xlsx", header=None)

print(f"Total rows in file: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nFirst few rows:")
print(df.head(10))

# Determine which column has commands
target_col = 1 if (str(df.iloc[0, 0]).isdigit() or "#" in str(df.iloc[0, 0])) else 0
print(f"\nUsing column {target_col} for commands")

# Extract all commands
commands = []
for idx, row in df.iterrows():
    cmd = str(row[target_col]).strip().lower()
    if cmd and cmd != 'nan' and not cmd.isdigit() and cmd != "#":
        commands.append(cmd)

print(f"\nExtracted {len(commands)} commands:\n")
for i, cmd in enumerate(commands, 1):
    print(f"{i:2d}. {cmd}")

# Save to a text file for easy reference
with open("extracted_commands.txt", "w", encoding="utf-8") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"\nCommands saved to 'extracted_commands.txt'")