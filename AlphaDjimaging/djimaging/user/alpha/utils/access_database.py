def access_database(indicator, use_sql=True):
    if use_sql:
        from IPython.display import clear_output
        if indicator == 'soma':
            from djimaging.user.alpha.utils import database_soma as database
            database.connect_dj()
        else:
            from djimaging.user.alpha.utils import database
            database.connect_dj(indicator=indicator)
        clear_output()
    else:
        if indicator == 'soma':
            from djimaging.user.alpha.utils.database_soma_no_sql import DatabaseInterface
            database = DatabaseInterface(indicator)
        else:
            from djimaging.user.alpha.utils.database_no_sql import DatabaseInterface
            database = DatabaseInterface(indicator)

    return database