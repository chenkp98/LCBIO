use strict;
use warnings;
use FindBin;
use Getopt::Long;

#=================================================
BEGIN {
	my $exec_bin="/public1/Programs/slurm/slurm_3.0";
	require("$exec_bin/config.pl");
	our ($Libs);
	print "slurm:$Libs\n";
	unshift(@INC, "$Libs");
	
}

use Slurm::Run;
#=================================================

my ($sh,$ls,$interval,$partition,$sd,$ed,$maxFailed,$maxProc,$cpu,$node,$gpu)=@ARGV;

# doInitialization();
runCMD();

#========================================================================

sub runCMD{
	my $runSlurm=Run->new("cmd"=>$sh,"ls"=>$ls,"interval"=>$interval,"partition"=>$partition,"sd"=>$sd,"ed"=>$ed,"maxFailed"=>$maxFailed,"maxProc"=>$maxProc,"cpu"=>$cpu,"node"=>$node,"gpu"=>$gpu)->runCmd();
	
	#0为正常退出，非0为非正常退出。
	if($runSlurm!=0){
		print STDERR "process in error. exiting...\n";
		exit -1;
	}
	else{
		print STDERR "process done.\n";
	}
	
}