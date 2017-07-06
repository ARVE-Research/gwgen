module parseghcnrow

contains

subroutine parse_station(infile, ndays, stationid, dates, variables, flags, j)

implicit none

character(11), intent(out) :: stationid

integer :: year
integer :: month

character(4) :: variable

character(300), intent(in) :: infile
integer, intent(in) :: ndays

! returns
integer, intent(out)       :: dates(ndays, 3), j
real, intent(out)          :: variables(ndays, 3)
character(1), intent(out)  :: flags(ndays, 3, 3)

integer, parameter, dimension(12) :: ndm_normal = [ 31,28,31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]  !number of days in each month
integer, parameter, dimension(12) :: ndm_leapyr = [ 31,29,31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]  !number of days in each month

integer, dimension(12) :: ndaymon

character(269) :: values

real,dimension(31) :: tmax
real,dimension(31) :: tmin
real,dimension(31) :: prcp

character(1), dimension(31,3) :: tmin_flags
character(1), dimension(31,3) :: tmax_flags
character(1), dimension(31,3) :: prcp_flags

integer :: d
integer :: start

integer :: lastmonth
integer :: lastyear

real, parameter :: missing = -9999.


!----

open(10,file=infile,status='old')

20 format(a11,i4,i2,a4,a247)

lastyear = 0
lastmonth = 0

do

    read(10,20,end=99)stationid,year,month,variable,values
    if (variable == 'PRCP' .or. variable == 'TMAX' .or. variable == 'TMIN') then

        !initialize variables

        lastmonth = month
        lastyear = year
        exit
    end if
end do

if (lastmonth == 0) then
    STOP 'Could not find any data for tmin, tmax or prcp'
end if

tmin = missing
tmax = missing
prcp = missing

tmin_flags = ''
tmax_flags = ''
prcp_flags = ''

rewind(10)

!----
j = 1
do !read one month of one variable per row

  read(10,20,end=99)stationid,year,month,variable,values

  if (variable /= 'PRCP' .and. variable /= 'TMAX' .and. variable /= 'TMIN') cycle

  if (year /= lastyear .or. month /= lastmonth) then

    !write out the data

    do d = 1,ndaymon(lastmonth)
      dates(j, :) = (/lastyear, lastmonth, d/)
      variables(j, :) = (/tmin(d), tmax(d), prcp(d)/)
      flags(j, 1, :) = tmin_flags(d,:)
      flags(j, 2, :) = tmax_flags(d,:)
      flags(j, 3, :) = prcp_flags(d,:)
      j = j + 1
    end do

    lastmonth = month
    lastyear  = year

    tmin = missing
    tmax = missing
    prcp = missing

    tmin_flags = ''
    tmax_flags = ''
    prcp_flags = ''

  end if

  if (leapyear(year)) then
    ndaymon = ndm_leapyr
  else
    ndaymon = ndm_normal
  end if

  do d = 1,ndaymon(month)

    start = 1+8*(d-1)

    !convert the value into a real number

    if (variable == 'TMAX') then

      read(values(start:start+4),*)tmax(d)
      !apply the scale factor
      tmax(d) = tmax(d) * 0.1
      ! Set values smaller than -100 degC to NaN
      if (tmax(d) < -100.0) tmax(d) = missing

      tmax_flags(d,1) = values(start+5:start+5)
      tmax_flags(d,2) = values(start+6:start+6)
      tmax_flags(d,3) = values(start+7:start+7)

    else if (variable == 'TMIN') then

      read(values(start:start+4),*)tmin(d)
      !apply the scale factor
      tmin(d) = tmin(d) * 0.1
      ! Set values smaller than -100 degC to NaN
      if (tmin(d) < -100.0) tmin(d) = missing

      tmin_flags(d,1) = values(start+5:start+5)
      tmin_flags(d,2) = values(start+6:start+6)
      tmin_flags(d,3) = values(start+7:start+7)

    else if (variable == 'PRCP') then

      read(values(start:start+4),*)prcp(d)
      !apply the scale factor
      prcp(d) = prcp(d) * 0.1
      if (prcp(d) < 0.0) prcp(d) = missing

      prcp_flags(d,1) = values(start+5:start+5)
      prcp_flags(d,2) = values(start+6:start+6)
      prcp_flags(d,3) = values(start+7:start+7)

    end if

  end do
end do

99 continue

if (lastmonth > 0) then
  do d = 1,ndaymon(lastmonth)
    dates(j, :) = (/lastyear, lastmonth, d/)
    variables(j, :) = (/tmin(d), tmax(d), prcp(d)/)
    flags(j, 1, :) = tmin_flags(d,:)
    flags(j, 2, :) = tmax_flags(d,:)
    flags(j, 3, :) = prcp_flags(d,:)
    j = j + 1
  end do
end if
j = j - 1

end subroutine parse_station

!----

logical function leapyear(year)

integer, intent(in) :: year

if ((mod(year,4) == 0 .and. mod(year,100) /= 0) .or. mod(year,400) == 0) then

  leapyear = .true.

else

  leapyear = .false.

end if

end function leapyear

!----

end module parseghcnrow
