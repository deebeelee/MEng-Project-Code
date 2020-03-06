import pandas
df = pandas.read_csv('dft_traffic_counts_raw_counts.csv')
print("done reading")
df = df[["count_point_id","direction_of_travel","year","count_date","hour"]].sort_values(
        by=['count_point_id',"count_date","hour"],kind='mergesort')
print("done sorting")
df.to_csv('sorted_by_id.csv',index=False)