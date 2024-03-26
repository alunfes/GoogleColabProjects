# Excelファイルを読み込み、指定したセルを基準にDataFrameを貼り付ける
wb = load_workbook(excel_file)
ws = wb[sheet_name]

start_row = 2  # DataFrameの貼り付けを開始する行
start_column = 2  # DataFrameの貼り付けを開始する列

for r_idx, row in enumerate(df.values):
    for c_idx, value in enumerate(row):
        cell = ws.cell(row=start_row + r_idx, column=start_column + c_idx)
        cell.value = value
