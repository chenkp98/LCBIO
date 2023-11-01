
use warnings;
use FindBin;
#==========require slurm==================
# BEGIN {
	# my $exec_bin="/public1/Programs/slurm";
	# require("$exec_bin/config.pl");
	# our ($Libs);
	
	# unshift(@INC, "$Libs");
# }
# use Slurm::Run;

my ($parameters_path, $partition, $cpu);
our $node = NA;

my ($min,$hour,$day,$month,$year)=(localtime)[1,2,3,4,5];
($min,$hour,$day,$month,$year) = (sprintf("%02d", $min),sprintf("%02d", $hour),sprintf("%02d", $day),sprintf("%02d", $month + 1),$year + 1900);

#our($bind_default);

#============Variable declaration===============
my $ML_dir="$FindBin::Bin";
require("$ML_dir/bin/config.pl");
our ($perl, $python);

## Container(软件、包环境)

#our($sif, $sif2,$bind_default);
our $ml = "$ML_dir/bin/ACGT101_ML_1.0.py";
require("$ML_dir/bin/sing_config.pl");
our ($sif,$bind_default);

my $maxProc=100000;
my $interval=50;
my $ls=0;#lineSplit.
my $maxFailed=0;
#====Log======================================================================
my $run_start_time="$year$month$day$hour$min";
my $run_cmd_log="run_cmd\_$run_start_time.log";
open(LOG,">$run_cmd_log") or die $!;
#集群配置参数
my $sd="sh_$run_start_time";#shell directory
my $ed="error";#error directory
#my $ld="log_$run_start_time";#log directory
system("rm -rf $ed $sd");
system("mkdir -p $sd $ed");
#################参数文件
my $parameter=$ARGV[0];
my @errors = get_parameters("$parameter");
if(exists ($errors[0])){
	foreach my $str(@errors){
		print STDERR"$str\n";
	}
	print STDERR"Quitting...\n";
	exit;
}

my $cmd="$python $ml -p $parameters_path";

my $job = "Start to run of ML program!";
my $sh_ML="$sd/sh_ML.sh";
print("$cmd\n");

print_cmd($cmd,$sh_ML,1,$node);
run_cmd($sh_ML,$ls,$interval,$partition,$sd,$ed,$maxFailed,$maxProc,1,$node);
#run_cmd($job,$sh_PPMS,1,$partition,$ed,$ld,$sd,1,$node);

# sub open_OUT1 {
	# my ($file) = @_;
	# my $OUT = ();
	# if($file=~/\.gz/){
		# open($OUT,"| gzip >$file") or die "could not write $file $!";
	# }else{
		# open($OUT,">$file") or die "could not write $file $!";
	# }
	
	# return $OUT;
# }	

sub print_cmd {
	my ($cmd,$out_sh,$cpu,$node) = @_;
	
	open(OUT,">>$out_sh");
	# if($cmd=~/^cd / || $cmd=~/^ln / || $cmd=~/^sed /){
	if($cmd=~/^cd / || $cmd=~/^ln /){
		print OUT "$cmd\n";
	}
	else{
		if($cmd=~/^#/){
			$cmd=~s/^#//g;
			if( defined($cpu) ){
				if ($node eq "NA"){
					print OUT "# srun -p $partition -c $cpu singularity exec --bind $bind_default $sif $cmd\n";
				}
				else{
					print OUT "# srun -p $partition -w $node -c $cpu singularity exec --bind $bind_default $sif $cmd\n";
				}
			}
			else{
				if ($node eq "NA"){
					print OUT "# srun -p $partition -c 1 singularity exec --bind $bind_default $sif $cmd\n";
				}
				else{
					print OUT "# srun -p $partition -w $node -c 1 singularity exec --bind $bind_default $sif $cmd\n";
				}
			}
		}
		else{
			if( defined($cpu) ){
				if ($node eq "NA"){
					print OUT "srun -p $partition -c $cpu singularity exec --bind $bind_default $sif $cmd\n";
				}
				else{
					print OUT "srun -p $partition -w $node -c $cpu singularity exec --bind $bind_default $sif $cmd\n";
				}
			}
			else{
				if ($node eq "NA"){
					print OUT "srun -p $partition -c 1 singularity exec --bind $bind_default $sif $cmd\n";
				}
				else{
					print OUT "srun -p $partition -w $node -c 1 singularity exec --bind $bind_default $sif $cmd\n";
				}
			}
		}
	}
	close OUT;
	# `sleep 0.1s`;
}

sub run_cmd {
	my ($sh,$ls,$interval,$partition,$sd,$ed,$maxFailed,$maxProc,$cpu,$node)=@_;
	#run_cmd($sh_PPMS,$ls,$interval,$partition,$sd,$ed,$maxFailed,$maxProc,1,$node);
	#slum.pl   my ($sh,$ls,$interval,$partition,$sd,$ed,$maxFailed,$maxProc,$cpu,$node,$gpu)=@ARGV;
	if(-e $sh){
		my ($start,$start_h) = get_start_time();
		my $ret = 0;
		my $cmd="$perl /public1/Programs/ACGT101_ML/ACGT101_ML_1.0/bin/slurm.pl $sh $ls $interval $partition $sd $ed $maxFailed $maxProc $cpu $node";
		
		print LOG "#$sh:\n";
		print LOG "$cmd\n";
		print STDERR "Start $sh at $start_h ...\n";
		
		$ret=system($cmd);
		
		my ($end,$end_h) = get_end_time();
		my $dur = get_duration($start,$end);
		my $status = "Done";
		if ($ret != 0) {
			$status = "Stop";
		}
		
		print LOG "#Run status: $status\n\n";
		print STDERR "$status at $end_h.\n";
		print STDERR "Duration: $dur\n\n";
		if ($ret != 0) {
			exit(-1);
		}
	}

}



sub get_parameters{
	my $paramfile_tmp=shift;
	my($key, $value,%paramfile_hash);
	my @faults=();
	open (INF, "$paramfile_tmp" ) || die "cannot open parameter file $paramfile_tmp: $!\n";
	while(<INF>){
		next if ($_ =~ /^\#/);
		chomp $_;
		s/^\s+//;
		s/\s+$//;
		s/\"//g;
		next if($_ eq "");
		my($key, $value) = split(/=/, $_);
		$key=~s/\s//g;
		$value=~s/\s+$//;
		$value=~s/^\s+//;
		$paramfile_hash{$key} = $value;
	}
	close INF;
	$parameters_path=$paramfile_hash{"parameters_path"};
	$partition=$paramfile_hash{"partition"};
	$cpu=$paramfile_hash{"cpu"};
	return @faults;
}


sub get_start_time {
	my $start = time();
	my $start_h = localtime();
	return ($start,$start_h);
}

sub get_end_time {
	my $end = time();
	my $end_h = localtime();
	return ($end,$end_h);
}


sub get_duration {#计算操作时间
	my ($start,$end)=@_;
	my $sec = $end - $start;
	my $days = int($sec/(24*60*60));
	my $hours = ($sec/(60*60))%24;
	my $mins = ($sec/60)%60;
	my $secs = $sec%60;
	
	return "$days days $hours hours $mins minutes $secs seconds";
}