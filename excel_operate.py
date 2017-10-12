metrics = "./metric.xlsx"
import xlrd
xl = xlrd.open_workbook(metrics)
sheet = xl.sheet_by_name('sheet')

# sheet.name
# sheet.nrows
# sheet.ncols
# sheet.cell_value(0,0)
# sheet.row_values(0)

metrics_df = []
for i in range(sheet.nrows):
    metrics_df.append(sheet.row_values(i))
metrics_df  = pd.DataFrame(metrics, columns=['name', 'result'])
