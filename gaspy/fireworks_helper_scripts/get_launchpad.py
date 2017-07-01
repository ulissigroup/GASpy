from fireworks import LaunchPad


def get_launchpad():
    ''' This function contains the information about our FireWorks LaunchPad '''
    return LaunchPad(host='mongodb01.nersc.gov',
                     name='fw_zu_vaspsurfaces',
                     username='admin_zu_vaspsurfaces',
                     password='$TPAHPmj',
                     port=27017)
