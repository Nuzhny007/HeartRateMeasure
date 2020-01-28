#pragma once
typedef float iir_float_t;

iir_float_t *binomial_mult( int n, iir_float_t *p );
iir_float_t *trinomial_mult( int n, iir_float_t *b, iir_float_t *c );

iir_float_t *dcof_bwlp( int n, iir_float_t fcf );
iir_float_t *dcof_bwhp( int n, iir_float_t fcf );
iir_float_t *dcof_bwbp( int n, iir_float_t f1f, iir_float_t f2f );
iir_float_t *dcof_bwbs( int n, iir_float_t f1f, iir_float_t f2f );

int *ccof_bwlp( int n );
int *ccof_bwhp( int n );
int *ccof_bwbp( int n );
iir_float_t *ccof_bwbs( int n, iir_float_t f1f, iir_float_t f2f );

iir_float_t sf_bwlp( int n, iir_float_t fcf );
iir_float_t sf_bwhp( int n, iir_float_t fcf );
iir_float_t sf_bwbp( int n, iir_float_t f1f, iir_float_t f2f );
iir_float_t sf_bwbs( int n, iir_float_t f1f, iir_float_t f2f );
