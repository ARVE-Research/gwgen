module parseeecra

use omp_lib

contains

subroutine parse_file( &
    infile, year, nrecords, &
    yr,mn,dy,hr,      & !   year,month,day,hour
    IB,               & !   sky brightness indicator
    LAT,              & !   latitude
    LON,              & !   longitude
    station_id,       & !   land: station number, ship: source deck, ship type
    LO,               & !   land/ocean indicator
    ww,               & !   present weather
    N,                & !   total cloud cover
    Nh,               & !   lower cloud amount
    h,                & !   lower cloud base height
    CL,               & !   low cloud type
    CM,               & !   middle cloud type
    CH,               & !   high cloud type
    AM,               & !   middle cloud amount
    AH,               & !   high cloud amount
    UM,               & !   middle cloud amount
    UH,               & !   high cloud amount
    IC,               & !   change code
    SA,               & !   solar altitude
    RI,               & !   relative lunar illuminance
    SLP,              & !   sea level pressure
    WS,               & !   wind speed
    WD,               & !   wind direction (degrees)
    AT,               & !   air temperature
    DD,               & !   dew point depression
    EL,               & !   Land: station elevation (m)
    IW,               & !   wind speed indicator
    IP)                 !  Land: sea level pressure flag
    !   year,month,day,hour          yr,mn,dy,hr    8   51120100  96113021    none
    !   sky brightness indicator        IB          1          0         1    none
    !   latitude  x100                  LAT         5      -9000      9000    none
    !   longitude x100                  LON         5          0     36000    none
    !   land: station number            ID          5      01000     98999    none
    !   ship: source deck, ship type                        1100      9999  none,9
    !   land/ocean indicator            LO          1          1         2    none
    !   present weather                 ww          2          0        99      -1
    !   total cloud cover               N           1          0         8    none
    !   lower cloud amount              Nh          2         -1         8      -1
    !   lower cloud base height         h           2         -1         9      -1
    !   low cloud type                  CL          2         -1        11      -1
    !   middle cloud type               CM          2         -1        12      -1
    !   high cloud type                 CH          2         -1         9      -1
    !   middle cloud amount x100        AM          3          0       800     900
    !   high cloud amount x100          AH          3          0       800     900
    !   middle cloud amount           UM          1          0         8       9
    !   high cloud amount             UH          1          0         8       9
    !   change code                     IC          2          0         9    none
    !   solar altitude (deg x10)        SA          4       -900       900    none
    !   relative lunar illuminance x100 RI          4       -110       117    none
    !   sea level pressure (mb x10)     SLP         5       9000,L   10999,L    -1
    !   wind speed (ms-1 x10)           WS          3          0       999      -1
    !   wind direction (degrees)        WD          3          0       361      -1
    !   air temperature (C x10)         AT          4       -949,L     599,L   900
    !   dew point depression (C x10)    DD          3          0       700     900
    !   Land: station elevation (m)     EL          4       -350      4877    9000
    !   wind speed indicator            IW          1          0         1       9
    !  Land: sea level pressure flag   IP          1          0)
    implicit none

    character(300), intent(in) :: infile
    integer, intent(in) :: nrecords, year
    integer, dimension(nrecords), intent(out) :: &
      yr,mn,dy,hr,      & !   year,month,day,hour
      IB,               & !   sky brightness indicator
      station_id,       & !   land: station number, ship: source deck, ship type
      LO,               & !   land/ocean indicator
      ww,               & !   present weather
      N,                & !   total cloud cover
      Nh,               & !   lower cloud amount
      h,                & !   lower cloud base height
      CL,               & !   low cloud type
      CM,               & !   middle cloud type
      CH,               & !   high cloud type
      UM,               & !   middle cloud amount
      UH,               & !   high cloud amount
      IC,               & !   change code
      WD,               & !   wind direction (degrees)
      EL,               & !   Land: station elevation (m)
      IW,               & !   wind speed indicator
      IP                  !  Land: sea level pressure flag
    real, dimension(nrecords), intent(out) :: &
      LAT,              & !   latitude
      LON,              & !   longitude
      AM,               & !   middle cloud amount
      AH,               & !   high cloud amount
      SA,               & !   solar altitude
      RI,               & !   relative lunar illuminance
      SLP,              & !   sea level pressure
      WS,               & !   wind speed
      AT,               & !   air temperature
      DD                  !   dew point depression
    integer :: i

    open(10,file=infile,status='old')

    20 format(i2,i2,i2,i2, & ! yr,mn,dy,hr
              i1,          & ! IB
              f5.0,          & ! LAT
              f5.0,          & ! LON
              i5,          & ! station_id
              i1,          & ! LO
              i2,          & ! ww
              i1,          & ! N
              i2,          & ! Nh
              i2,          & ! h
              i2,          & ! CL
              i2,          & ! CM
              i2,          & ! CH
              f3.0,          & ! AM
              f3.0,          & ! AH
              i1,          & ! UM
              i1,          & ! UH
              i2,          & ! IC
              f4.0,          & ! SA
              f4.0,          & ! RI
              f5.0,          & ! SLP
              f3.0,          & ! WS
              i3,          & ! WD
              f4.0,          & ! AT
              f3.0,          & ! DD
              i4,          & ! EL
              i1,          & ! IW
              i1)            ! IP
    21 format(i4,i2,i2,i2, & ! yr,mn,dy,hr
              i1,          & ! IB
              f5.0,          & ! LAT
              f5.0,          & ! LON
              i5,          & ! station_id
              i1,          & ! LO
              i2,          & ! ww
              i1,          & ! N
              i2,          & ! Nh
              i2,          & ! h
              i2,          & ! CL
              i2,          & ! CM
              i2,          & ! CH
              f3.0,          & ! AM
              f3.0,          & ! AH
              i1,          & ! UM
              i1,          & ! UH
              i2,          & ! IC
              f4.0,          & ! SA
              f4.0,          & ! RI
              f5.0,          & ! SLP
              f3.0,          & ! WS
              i3,          & ! WD
              f4.0,          & ! AT
              f3.0,          & ! DD
              i4,          & ! EL
              i1,          & ! IW
              i1)            ! IP
    do i=1,nrecords
        if (year < 1997) then
            read(10,20) yr(i),mn(i),dy(i),hr(i),IB(i),LAT(i),LON(i),station_id(i), &
                       LO(i),ww(i),N(i),Nh(i),h(i),CL(i),CM(i),CH(i),AM(i),AH(i), &
                       UM(i),UH(i),IC(i),SA(i),RI(i),SLP(i),WS(i),WD(i),AT(i),DD(i), &
                       EL(i),IW(i),IP(i)
            yr(i) = yr(i) + 1900
        else
            read(10,21) yr(i),mn(i),dy(i),hr(i),IB(i),LAT(i),LON(i),station_id(i), &
                       LO(i),ww(i),N(i),Nh(i),h(i),CL(i),CM(i),CH(i),AM(i),AH(i), &
                       UM(i),UH(i),IC(i),SA(i),RI(i),SLP(i),WS(i),WD(i),AT(i),DD(i), &
                       EL(i),IW(i),IP(i)
        endif
        if (LON(i) > 18000) LON(i) = LON(i) - 36000
        LAT(i) = LAT(i) * 0.01
        LON(i) = LON(i) * 0.01
        AM(i) = AM(i) * 0.01
        AH(i) = AH(i) * 0.01
        SA(i) = SA(i) * 0.01
        RI(i) = RI(i) * 0.01
        SLP(i) = SLP(i) * 0.01
        WS(i) = WS(i) * 0.1
        AT(i) = AT(i) * 0.1
        DD(i) = DD(i) * 0.1
    end do

end subroutine parse_file

subroutine extract_data(ids, src_dir, odir, years, imonths)

    implicit none

    integer, intent(in), dimension(:) :: ids, years, imonths
    character*300, intent(in) :: src_dir, odir

    INTEGER :: &
        yr,mn,dy,hr,      & !   year,month,day,hour          yr,mn,dy,hr    8   51120100  96113021    none
        IB,               & !   sky brightness indicator        IB          1          0         1    none
        LAT,              & !   latitude  x100                  LAT         5      -9000      9000    none
        LON,              & !   longitude x100                  LON         5          0     36000    none
        station_id,       & !   land: station number            ID          5      01000     98999    none
                            !   ship: source deck, ship type                        1100      9999  none,9
        LO,               & !   land/ocean indicator            LO          1          1         2    none
        ww,               & !   present weather                 ww          2          0        99      -1
        N,                & !   total cloud cover               N           1          0         8    none
        Nh,               & !   lower cloud amount              Nh          2         -1         8      -1
        h,                & !   lower cloud base height         h           2         -1         9      -1
        CL,               & !   low cloud type                  CL          2         -1        11      -1
        CM,               & !   middle cloud type               CM          2         -1        12      -1
        CH,               & !   high cloud type                 CH          2         -1         9      -1
        AM,               & !   middle cloud amount x100        AM          3          0       800     900
        AH,               & !   high cloud amount x100          AH          3          0       800     900
        UM,               & !   middle cloud amount           UM          1          0         8       9
        UH,               & !   high cloud amount             UH          1          0         8       9
        IC,               & !   change code                     IC          2          0         9    none
        SA,               & !   solar altitude (deg x10)        SA          4       -900       900    none
        RI,               & !   relative lunar illuminance x100 RI          4       -110       117    none
        SLP,              & !   sea level pressure (mb x10)     SLP         5       9000,L   10999,L    -1
        WS,               & !   wind speed (ms-1 x10)           WS          3          0       999      -1
        WD,               & !   wind direction (degrees)        WD          3          0       361      -1
        AT,               & !   air temperature (C x10)         AT          4       -949,L     599,L   900
        DD,               & !   dew point depression (C x10)    DD          3          0       700     900
        EL,               & !   Land: station elevation (m)     EL          4       -350      4877    9000
        IW,               & !   wind speed indicator            IW          1          0         1       9
        IP                  !  Land: sea level pressure flag   IP          1          0         2       9

    integer(kind = OMP_lock_kind) :: lck = 0
    integer(kind = OMP_lock_kind), dimension(size(ids)) :: locks
    integer :: ID
    integer :: j = 0, i = 0, nids = 0, keep = 0, old_id = 0, year = 0, &
               imonth = 0, iyear = 0
    integer, dimension(size(ids)) :: ounits
    integer, dimension(100) :: inunits = 0
    character*350, dimension(size(ids)) :: ofiles
    character*90 :: cfmt
    character*300 :: all_cfmt
    character*350 :: fname
    character*10 :: cid
    character*2 :: cyear
    character(len=3), dimension(12) :: months = (/ &
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', &
        'NOV', 'DEC' /)
    logical :: file_exists

    !       mn,dy,hr,IB,la,lo,id,LO,ww,N ,Nh,h ,CL,CM,CH,AM,AH,UM,UH,IC,SA,RI,SLP,WS,WD,AT,DD,EL,IW,IP
    cfmt = 'i2,i2,i2,i1,i5,i5,i5,i1,i2,i1,i2,i2,i2,i2,i2,i3,i3,i1,i1,i2,i4,i4,i5,i3 ,i3,i4,i3,i4,i1,i1'

    ! output format
    25 format(i4,a,i0.2,a,i0.2,a,i0.2, & ! yr,mn,dy,hr
              a,           &
              i1,          & ! IB
              a,           &
              f6.2,        & ! LAT
              a,           &
              f6.2,        & ! LON
              a,           &
              i5,          & ! station_id
              a,           &
              i1,          & ! LO
              a,           &
              i2,          & ! ww
              a,           &
              i1,          & ! N
              a,           &
              i2,          & ! Nh
              a,           &
              i2,          & ! h
              a,           &
              i2,          & ! CL
              a,           &
              i2,          & ! CM
              a,           &
              i2,          & ! CH
              a,           &
              f4.2,        & ! AM
              a,           &
              f5.2,        & ! AH
              a,           &
              i1,          & ! UM
              a,           &
              i1,          & ! UH
              a,           &
              i2,          & ! IC
              a,           &
              f5.1,        & ! SA
              a,           &
              f5.2,        & ! RI
              a,           &
              f6.1,        & ! SLP
              a,           &
              f4.1,        & ! WS
              a,           &
              i3,          & ! WD
              a,           &
              f5.1,        & ! AT
              a,           &
              f4.1,        & ! DD
              a,           &
              i4,          & ! EL
              a,           &
              i1,          & ! IW
              a,           &
              i1)            ! IP

nids = size(ids)

j = 11
do i = 1, nids
    locks(i) = i
    call OMP_init_lock(locks(i))
    write(cid, '(I10)') ids(i)
    ofiles(i) = trim(odir)//trim(adjustl(cid))//'.csv'
    ounits(i) = j + 1000
    open(10, file=trim(ofiles(i)), status='unknown',action='write')
    write(10,'(a)') 'year,month,day,hour,IB,lat,lon,station_id,LO,ww,N,Nh,h,CL,CM,CH,AM,AH,UM,UH,IC,SA,RI,SLP,WS,WD,AT,DD,EL,IW,IP'
    j = j + 1
    close(10, status='keep')
end do

call OMP_init_lock(lck)

do i = 1,size(inunits)
    inunits(i) = i + 50
end do

!$OMP PARALLEL DO SHARED(lck,locks,src_dir,odir,nids,ids,months,inunits,ounits, &
!$OMP&                  ofiles,cfmt,imonths,years), &
!$OMP FIRSTPRIVATE(old_id,keep), &
!$OMP& PRIVATE(yr,mn,dy,hr,IB,LAT,LON,station_id,LO,ww,N,Nh, h,CL,CM,CH,AM,AH, &
!$OMP&         UM,UH,IC,SA,RI,SLP,WS,WD,AT, DD,EL,IW,IP,fname,cid,i,ID,imonth, &
!$OMP&         cyear,year,all_cfmt,iyear,j)
do iyear = 1,size(years)
    year = years(iyear)
    ID = OMP_get_thread_num() + 1
    call OMP_set_lock(lck)
    write(*,*) "My thread is ", ID, '. Processing year ', year
    call OMP_unset_lock(lck)
    if (year < 1997) then
        all_cfmt = '(i2,'//cfmt//')'
    else
        all_cfmt = '(i4,'//cfmt//')'
    endif
    do imonth = 1,size(imonths)
        write(cyear,'(I0.2)') mod(year, 100)
        fname = trim(src_dir)//months(imonths(imonth))//cyear//'L'
        inquire(file=fname,EXIST=file_exists)
        if (.not. file_exists) goto 99
        open(inunits(ID), file=fname, status='old', action='read')
        j = 1
        do
            read(inunits(ID),all_cfmt,end=99,err=98) yr,mn,dy,hr,IB,LAT,LON, &
                               station_id,LO,ww,N,Nh,h,CL,CM,CH,AM,AH,UM,UH, &
                               IC,SA,RI,SLP,WS,WD,AT,DD,EL,IW,IP
            if (year < 1997) yr = yr + 1900
            if (old_id /= station_id) then
                if (keep /= 0) then
                    close(ounits(keep))
                    call OMP_unset_lock(locks(keep))
                endif
                keep = 0
                do i = 1, nids
                    if (station_id == ids(i)) then
                        keep = i
                        call OMP_set_lock(locks(keep))
                        write(cid, '(I10)') ids(i)
                        open(ounits(keep), file=trim(ofiles(keep)), &
                             status='old',access='append',action='write')
                    endif
                end do
                old_id = station_id
            endif
            if (keep > 0) then
                if (LON > 18000) LON = LON - 36000
                write(ounits(keep),25) yr,',',mn,',',dy,',',hr,',',IB,',', &
                             0.01*LAT,',',0.01*LON,',',station_id,',', &
                             LO,',',ww,',',N,',',Nh,',',h,',',CL,',', &
                             CM,',',CH,',',0.01*AM,',',0.01*AH,',',UM, &
                             ',',UH,',',IC,',',0.1*SA,',',0.01*RI,',', &
                             0.1*SLP,',',0.1*WS,',',WD,',',0.1*AT,',', &
                             0.1*DD,',',EL,',',IW,',',IP
            end if
            j = j + 1
            goto 97
98 continue
            write(*,*) "Error occured at line ", j, "of file ", fname
97 continue
        end do
99 continue
    close(inunits(ID))
    if (keep /= 0) then
        close(ounits(keep))
        call OMP_unset_lock(locks(keep))
        keep = 0
    endif
    end do
end do
!$OMP END PARALLEL DO

call OMP_destroy_lock(lck)
do i=1,nids
    call OMP_destroy_lock(locks(i))
end do
end subroutine extract_data

end module parseeecra
