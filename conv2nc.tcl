#! ~jecole/bin/Convsh1.94

#  Convsh script conv2nc.tcl
#
#  Convert input files into single netCDF file.
#  All input files must contain the same fields and have
#  identical dimensions except for the time dimension.
#  For example to convert UM output files into a netCDF file
#  use the following command:
#
#      ./conv2nc.tcl -i xaavaa.pc* -o xaava.nc -f 148 149

#  Write out netCDF file
set outformat netcdf

#  Automatically work out input file type
set filetype 0

#  Get command line arguments:
#      -i input files (can be more than one file)
#      -o output file (single file only)
#      -f field list (defaults to all fields)

set i false
set o false
set f false
foreach arg $argv {
   switch -glob -- $arg {
      -i      {set i true ; set o false ; set f false}
      -o      {set i false ; set o true ; set f false}
      -f      {set i false ; set o false ; set f true}
      -*      {puts "unknown option $arg" ; set i false; set o false ; set f false}
      default {
         if {$i} {
            set infile [lappend infile $arg]
         } elseif {$o} {
            set outfile [lappend outfile $arg]
         } elseif {$f} {
            set fieldlist [lappend fieldlist $arg]
         } else {
            puts "unknown option $arg"
         }
      }
   }
}

if {! [info exists infile]} {
   puts "input file name must be given"
   exit
}

if {[info exists outfile]} {
   if {[llength $outfile] > 1} {
      set outfile [lindex $outfile 0]
      puts "Only one output file can be specified, using $outfile"
   }
} else {
   puts "output file name must be given"
   exit
}

if {! [info exists fieldlist]} {
   #  Convert all fields in input files to netCDF
   set fieldlist -1
}


#  Read in each of the input files
foreach file $infile {
   readfile $filetype $file
}

#  Write out all input fields to a single netCDF file
writefile $outformat $outfile $fieldlist