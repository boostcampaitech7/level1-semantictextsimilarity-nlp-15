# import pandas as pd
# import os
# from datetime import datetime, timedelta, timezone
#
# def ensemble_predictions(output_dir, save_path):
#     # List all CSV files in the output directory
#     csv_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')]
#
#     # Ensure there are exactly 4 CSV files
#     if len(csv_files) != 4:
#         raise ValueError("There must be exactly 4 CSV files in the output directory.")
#
#     # Read all CSV files into dataframes
#     dataframes = [pd.read_csv(f) for f in csv_files]
#
#     # Ensure all dataframes have the same structure
#     for df in dataframes[1:]:
#         if not dataframes[0].columns.equals(df.columns):
#             raise ValueError("All CSV files must have the same structure.")
#
#     # Combine predictions using a 1:1:1:1 ratio
#     combined_predictions = (
#         1 * dataframes[0]['target'] +
#         1 * dataframes[1]['target'] +
#         1 * dataframes[2]['target'] +
#         1 * dataframes[3]['target']
#     ) / 4.0  # Divide by the total weight (1+1+1+1=4)
#
#     # Create the output dataframe
#     output_df = dataframes[0].copy()
#     output_df['target'] = combined_predictions.round(1)
#
#     # Save the combined predictions to output_날짜_시간.csv
#     kst = timezone(timedelta(hours=9))  # 한국 표준시 (UTC+9)
#     now = datetime.now(kst)
#     timestamp = now.strftime("%y%m%d_%H%M")
#     save_path = os.path.join(output_dir, f'output_{timestamp}.csv')
#
#     output_df.to_csv(save_path, index=False)
#
# def main():
#     output_dir = 'level1-semantictextsimilarity-nlp-15/src/csv_ensemble'  # Specify the output directory
#     ensemble_predictions(output_dir, None)  # save_path is now handled inside the function
#
# if __name__ == "__main__":
#     main()
