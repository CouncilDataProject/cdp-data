from cdp_data import CDPInstances, datasets


def voting_analysis(cdp_instance:str, start_datetime:str, end_datetime:str,):
    ds = datasets.get_vote_dataset(cdp_instance,start_datetime=start_datetime,end_datetime=end_datetime,replace_py_objects=True)

    ds=ds[['matter_id','event_datetime','event_minutes_item_overall_decision','in_majority','decision','person_name']].sort_values(by=['matter_id'])

    #all of the voting in the system
    ds_all_votes = ds.groupby(['matter_id','event_datetime','event_minutes_item_overall_decision'])

    #all of the votings that passed
    df_passed = ds_all_votes['event_minutes_item_overall_decision'].apply(lambda x: (x == 'Passed').sum()).reset_index(name='count')

    #all of voting that were not unanimous
    df_count_objection = ds.groupby(['matter_id'])['in_majority'].apply(lambda x: (x == False).sum()).reset_index(name='count')
    df_count_objection = df_count_objection[df_count_objection['count'] != 0]
#    print(ds_all_votes)
#    print(df_passed)
#    print(df_count_objection)
    return (len(ds_all_votes),len(df_passed),len(df_count_objection))


"""
f = open("council_vote_data.txt", "a")


for instance in CDPInstances.all_instances:
    res = voting_analysis(instance, "2022-10-01", "2022-10-31")
    f.write(instance + "\t" + str(res[0]) + "\t" + str(res[1]) + "\t" + str(res[2]))
    f.write("\n")

f.close()
"""

res = voting_analysis(CDPInstances.Oakland, "2022-10-01", "2022-10-30")
print(res)
