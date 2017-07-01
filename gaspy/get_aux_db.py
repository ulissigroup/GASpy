from vasp.mongo import MongoDatabase

def get_aux_db():
    ''' This is the information for the Auxiliary vasp.mongo database '''
    return MongoDatabase(host='mongodb01.nersc.gov',
                         port=27017,
                         user='admin_zu_vaspsurfaces',
                         password='$TPAHPmj',
                         database='vasp_zu_vaspsurfaces',
                         collection='atoms')
