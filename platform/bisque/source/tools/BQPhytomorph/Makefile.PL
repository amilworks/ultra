use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME                => 'BQPhytomorph',
    AUTHOR              => 'CBI <bisque-dev@biodev.example.com>',
    VERSION_FROM        => 'lib/BQPhytomorph.pm',
    ABSTRACT_FROM       => 'lib/BQPhytomorph.pm',
    PL_FILES            => {},
    PREREQ_PM => {
        'Test::More' => 0,
    },
    dist                => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
    clean               => { FILES => 'BQPhytomorph-*' },
);
