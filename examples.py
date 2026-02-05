import refscancsm

cpx_path = '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/tempo/CoilsmapFromCPX/patient47/test_nopatch/senserefscan/an_19112024_1620384_1000_2_senserefscanV4.cpx'
sin_path_refscan = '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/tempo/CoilsmapFromCPX/patient47/test_nopatch/an_19112024_1620384_1000_2_senserefscanV4.sin'
sin_path_target = '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/tempo/CoilsmapFromCPX/patient47/test_nopatch/an_19112024_1625290_3_2_tempo_cst1_gdV4.sin'


# csm_source = refscancsm._load_source_csm_from_cpx(cpx_path);

csm = refscancsm.get_csm(cpx_path, sin_path_refscan, sin_path_target);


