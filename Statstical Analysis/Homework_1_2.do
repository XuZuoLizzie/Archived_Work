import excel "\\tsclient\Drives\PH1700\boneden.xls", sheet("bone2") firstrow

generate A = ls2 - ls1

generate B = (ls1 + ls2)/2

generate C = 100 * A/B

generate diff_pyr = abs(pyr1 - pyr2)

generate cat_pyr = . 
replace cat_pyr = 1 if diff_pyr <= 9.9
replace cat_pyr = 2 if diff_pyr > 9.9 & diff_pyr <= 19.9
replace cat_pyr = 3 if diff_pyr > 19.9 & diff_pyr <= 29.9
replace cat_pyr = 4 if diff_pyr > 29.9 & diff_pyr <= 39.9
replace cat_pyr = 5 if diff_pyr > 39.9

label define pyr_label 1 "0-9.9" 2 "10-19.9" 3 "20-29.9" 4 "30-39.9" 5 "40+"
label values cat_pyr pyr_label

summarize C
tabstat C, statistics( count mean sd min max ) by(cat_pyr)
graph twoway (scatter C cat_pyr)