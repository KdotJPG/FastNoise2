#include "Simplex.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Simplex, SIMD> final : public virtual FastNoise::Simplex, public FastSIMD::DispatchClass<FastNoise::ScalableGenerator, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        this->ScalePositions( x, y );

        const float SQRT3 = 1.7320508075688772935274463415059f;
        const float F2 = 0.5f * (SQRT3 - 1.0f);
        const float G2 = (3.0f - SQRT3) / 6.0f;

        float32v f = float32v( F2 ) * (x + y);
        float32v x0 = FS::Floor( x + f );
        float32v y0 = FS::Floor( y + f );

        int32v i = FS::Convert<int32_t>( x0 ) * int32v( Primes::X );
        int32v j = FS::Convert<int32_t>( y0 ) * int32v( Primes::Y );

        float32v g = float32v( G2 ) * (x0 + y0);
        x0 = x - (x0 - g);
        y0 = y - (y0 - g);

        mask32v i1 = x0 > y0;
        //mask32v j1 = ~i1; //InvMasked funcs

        float32v x1 = FS::MaskedSub( i1, x0, float32v( 1.f ) ) + float32v( G2 );
        float32v y1 = FS::InvMaskedSub( i1, y0, float32v( 1.f ) ) + float32v( G2 );

        float32v x2 = x0 + float32v( G2 * 2 - 1 );
        float32v y2 = y0 + float32v( G2 * 2 - 1 );

        float32v t0 = FS::FNMulAdd( x0, x0, FS::FNMulAdd( y0, y0, float32v( 0.5f ) ) );
        float32v t1 = FS::FNMulAdd( x1, x1, FS::FNMulAdd( y1, y1, float32v( 0.5f ) ) );
        float32v t2 = FS::FNMulAdd( x2, x2, FS::FNMulAdd( y2, y2, float32v( 0.5f ) ) );

        t0 = FS::Max( t0, float32v( 0 ) );
        t1 = FS::Max( t1, float32v( 0 ) );
        t2 = FS::Max( t2, float32v( 0 ) );

        t0 *= t0; t0 *= t0;
        t1 *= t1; t1 *= t1;
        t2 *= t2; t2 *= t2;

        float32v n0 = GetGradientDotFancy( HashPrimes( seed, i, j ), x0, y0 );
        float32v n1 = GetGradientDotFancy( HashPrimes( seed, FS::MaskedAdd( i1, i, int32v( Primes::X ) ), FS::InvMaskedAdd( i1, j, int32v( Primes::Y ) ) ), x1, y1 );
        float32v n2 = GetGradientDotFancy( HashPrimes( seed, i + int32v( Primes::X ), j + int32v( Primes::Y ) ), x2, y2 );

        return float32v( 38.283687591552734375f ) * FS::FMulAdd( n0, t0, FS::FMulAdd( n1, t1, n2 * t2 ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        this->ScalePositions( x, y, z );

        const float F3 = 1.0f / 3.0f;
        const float G3 = 1.0f / 2.0f;

        float32v s = float32v( F3 ) * (x + y + z);
        x += s;
        y += s;
        z += s;

        float32v x0 = FS::Floor( x );
        float32v y0 = FS::Floor( y );
        float32v z0 = FS::Floor( z );
        float32v xi = x - x0;
        float32v yi = y - y0;
        float32v zi = z - z0;

        int32v i = FS::Convert<int32_t>( x0 ) * int32v( Primes::X );
        int32v j = FS::Convert<int32_t>( y0 ) * int32v( Primes::Y );
        int32v k = FS::Convert<int32_t>( z0 ) * int32v( Primes::Z );

        mask32v x_ge_y = xi >= yi;
        mask32v y_ge_z = yi >= zi;
        mask32v x_ge_z = xi >= zi;

        float32v g = float32v( G3 ) * (xi + yi + zi);
        x0 = xi - g;
        y0 = yi - g;
        z0 = zi - g;

        mask32v i1 = x_ge_y & x_ge_z;
        mask32v j1 = FS::BitwiseAndNot( y_ge_z, x_ge_y );
        mask32v k1 = FS::BitwiseAndNot( ~x_ge_z, y_ge_z );

        mask32v i2 = x_ge_y | x_ge_z;
        mask32v j2 = ~x_ge_y | y_ge_z;
        mask32v k2 = x_ge_z & y_ge_z; //InvMasked

        float32v x1 = FS::MaskedSub( i1, x0, float32v( 1 ) ) + float32v( G3 );
        float32v y1 = FS::MaskedSub( j1, y0, float32v( 1 ) ) + float32v( G3 );
        float32v z1 = FS::MaskedSub( k1, z0, float32v( 1 ) ) + float32v( G3 );
        float32v x2 = FS::MaskedSub( i2, x0, float32v( 1 ) ) + float32v( G3 * 2 );
        float32v y2 = FS::MaskedSub( j2, y0, float32v( 1 ) ) + float32v( G3 * 2 );
        float32v z2 = FS::InvMaskedSub( k2, z0, float32v( 1 ) ) + float32v( G3 * 2 );
        float32v x3 = x0 + float32v( G3 * 3 - 1 );
        float32v y3 = y0 + float32v( G3 * 3 - 1 );
        float32v z3 = z0 + float32v( G3 * 3 - 1 );

        float32v t0 = FS::FNMulAdd( x0, x0, FS::FNMulAdd( y0, y0, FS::FNMulAdd( z0, z0, float32v( 0.6f ) ) ) );
        float32v t1 = FS::FNMulAdd( x1, x1, FS::FNMulAdd( y1, y1, FS::FNMulAdd( z1, z1, float32v( 0.6f ) ) ) );
        float32v t2 = FS::FNMulAdd( x2, x2, FS::FNMulAdd( y2, y2, FS::FNMulAdd( z2, z2, float32v( 0.6f ) ) ) );
        float32v t3 = FS::FNMulAdd( x3, x3, FS::FNMulAdd( y3, y3, FS::FNMulAdd( z3, z3, float32v( 0.6f ) ) ) );

        t0 = FS::Max( t0, float32v( 0 ) );
        t1 = FS::Max( t1, float32v( 0 ) );
        t2 = FS::Max( t2, float32v( 0 ) );
        t3 = FS::Max( t3, float32v( 0 ) );

        t0 *= t0; t0 *= t0;
        t1 *= t1; t1 *= t1;
        t2 *= t2; t2 *= t2;
        t3 *= t3; t3 *= t3;             

        float32v n0 = GetGradientDotFancy( HashPrimes( seed, i, j, k ), x0, y0, z0 );
        float32v n1 = GetGradientDotFancy( HashPrimes( seed, FS::MaskedAdd( i1, i, int32v( Primes::X ) ), FS::MaskedAdd( j1, j, int32v( Primes::Y ) ), FS::MaskedAdd( k1, k, int32v( Primes::Z ) ) ), x1, y1, z1 );
        float32v n2 = GetGradientDotFancy( HashPrimes( seed, FS::MaskedAdd( i2, i, int32v( Primes::X ) ), FS::MaskedAdd( j2, j, int32v( Primes::Y ) ), FS::InvMaskedAdd( k2, k, int32v( Primes::Z ) ) ), x2, y2, z2 );
        float32v n3 = GetGradientDotFancy( HashPrimes( seed, i + int32v( Primes::X ), j + int32v( Primes::Y ), k + int32v( Primes::Z ) ), x3, y3, z3 );

        return float32v( 32.69428253173828125f ) * FS::FMulAdd( n0, t0, FS::FMulAdd( n1, t1, FS::FMulAdd( n2, t2, n3 * t3 ) ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        this->ScalePositions( x, y, z, w );

        const float SQRT5 = 2.236067977499f;
        const float F4 = (SQRT5 - 1.0f) / 4.0f;
        const float G4 = (5.0f - SQRT5) / 20.0f;

        float32v s = float32v( F4 ) * (x + y + z + w);
        x += s;
        y += s;
        z += s;
        w += s;

        float32v x0 = FS::Floor( x );
        float32v y0 = FS::Floor( y );
        float32v z0 = FS::Floor( z );
        float32v w0 = FS::Floor( w );
        float32v xi = x - x0;
        float32v yi = y - y0;
        float32v zi = z - z0;
        float32v wi = w - w0;

        int32v i = FS::Convert<int32_t>( x0 ) * int32v( Primes::X );
        int32v j = FS::Convert<int32_t>( y0 ) * int32v( Primes::Y );
        int32v k = FS::Convert<int32_t>( z0 ) * int32v( Primes::Z );
        int32v l = FS::Convert<int32_t>( w0 ) * int32v( Primes::W );

        float32v g = float32v( G4 ) * (xi + yi + zi + wi);
        x0 = xi - g;
        y0 = yi - g;
        z0 = zi - g;
        w0 = wi - g;

        int32v rankx( 0 );
        int32v ranky( 0 );
        int32v rankz( 0 );
        int32v rankw( 0 );

        mask32v x_ge_y = x0 >= y0;
        rankx = FS::MaskedIncrement( x_ge_y, rankx );
        ranky = FS::MaskedIncrement( ~x_ge_y, ranky );

        mask32v x_ge_z = x0 >= z0;
        rankx = FS::MaskedIncrement( x_ge_z, rankx );
        rankz = FS::MaskedIncrement( ~x_ge_z, rankz );

        mask32v x_ge_w = x0 >= w0;
        rankx = FS::MaskedIncrement( x_ge_w, rankx );
        rankw = FS::MaskedIncrement( ~x_ge_w, rankw );

        mask32v y_ge_z = y0 >= z0;
        ranky = FS::MaskedIncrement( y_ge_z, ranky );
        rankz = FS::MaskedIncrement( ~y_ge_z, rankz );

        mask32v y_ge_w = y0 >= w0;
        ranky = FS::MaskedIncrement( y_ge_w, ranky );
        rankw = FS::MaskedIncrement( ~y_ge_w, rankw );

        mask32v z_ge_w = z0 >= w0;
        rankz = FS::MaskedIncrement( z_ge_w, rankz );
        rankw = FS::MaskedIncrement( ~z_ge_w, rankw );

        mask32v i1 = rankx > int32v( 2 );
        mask32v j1 = ranky > int32v( 2 );
        mask32v k1 = rankz > int32v( 2 );
        mask32v l1 = rankw > int32v( 2 );

        mask32v i2 = rankx > int32v( 1 );
        mask32v j2 = ranky > int32v( 1 );
        mask32v k2 = rankz > int32v( 1 );
        mask32v l2 = rankw > int32v( 1 );

        mask32v i3 = rankx > int32v( 0 );
        mask32v j3 = ranky > int32v( 0 );
        mask32v k3 = rankz > int32v( 0 );
        mask32v l3 = rankw > int32v( 0 );

        float32v x1 = FS::MaskedSub( i1, x0, float32v( 1 ) ) + float32v( G4 );
        float32v y1 = FS::MaskedSub( j1, y0, float32v( 1 ) ) + float32v( G4 );
        float32v z1 = FS::MaskedSub( k1, z0, float32v( 1 ) ) + float32v( G4 );
        float32v w1 = FS::MaskedSub( l1, w0, float32v( 1 ) ) + float32v( G4 );
        float32v x2 = FS::MaskedSub( i2, x0, float32v( 1 ) ) + float32v( G4 * 2 );
        float32v y2 = FS::MaskedSub( j2, y0, float32v( 1 ) ) + float32v( G4 * 2 );
        float32v z2 = FS::MaskedSub( k2, z0, float32v( 1 ) ) + float32v( G4 * 2 );
        float32v w2 = FS::MaskedSub( l2, w0, float32v( 1 ) ) + float32v( G4 * 2 );
        float32v x3 = FS::MaskedSub( i3, x0, float32v( 1 ) ) + float32v( G4 * 3 );
        float32v y3 = FS::MaskedSub( j3, y0, float32v( 1 ) ) + float32v( G4 * 3 );
        float32v z3 = FS::MaskedSub( k3, z0, float32v( 1 ) ) + float32v( G4 * 3 );
        float32v w3 = FS::MaskedSub( l3, w0, float32v( 1 ) ) + float32v( G4 * 3 );
        float32v x4 = x0 + float32v( G4 * 4 - 1 );
        float32v y4 = y0 + float32v( G4 * 4 - 1 );
        float32v z4 = z0 + float32v( G4 * 4 - 1 );
        float32v w4 = w0 + float32v( G4 * 4 - 1 );

        float32v t0 = FS::FNMulAdd( x0, x0, FS::FNMulAdd( y0, y0, FS::FNMulAdd( z0, z0, FS::FNMulAdd( w0, w0, float32v( 0.6f ) ) ) ) );
        float32v t1 = FS::FNMulAdd( x1, x1, FS::FNMulAdd( y1, y1, FS::FNMulAdd( z1, z1, FS::FNMulAdd( w1, w1, float32v( 0.6f ) ) ) ) );
        float32v t2 = FS::FNMulAdd( x2, x2, FS::FNMulAdd( y2, y2, FS::FNMulAdd( z2, z2, FS::FNMulAdd( w2, w2, float32v( 0.6f ) ) ) ) );
        float32v t3 = FS::FNMulAdd( x3, x3, FS::FNMulAdd( y3, y3, FS::FNMulAdd( z3, z3, FS::FNMulAdd( w3, w3, float32v( 0.6f ) ) ) ) );
        float32v t4 = FS::FNMulAdd( x4, x4, FS::FNMulAdd( y4, y4, FS::FNMulAdd( z4, z4, FS::FNMulAdd( w4, w4, float32v( 0.6f ) ) ) ) );

        t0 = FS::Max( t0, float32v( 0 ) );
        t1 = FS::Max( t1, float32v( 0 ) );
        t2 = FS::Max( t2, float32v( 0 ) );
        t3 = FS::Max( t3, float32v( 0 ) );
        t4 = FS::Max( t4, float32v( 0 ) );

        t0 *= t0; t0 *= t0;
        t1 *= t1; t1 *= t1;
        t2 *= t2; t2 *= t2;
        t3 *= t3; t3 *= t3;
        t4 *= t4; t4 *= t4;

        float32v n0 = GetGradientDotFancy( HashPrimes( seed, i, j, k, l ), x0, y0, z0, w0 );
        float32v n1 = GetGradientDotFancy( HashPrimes( seed,
            FS::MaskedAdd( i1, i, int32v( Primes::X ) ),
            FS::MaskedAdd( j1, j, int32v( Primes::Y ) ),
            FS::MaskedAdd( k1, k, int32v( Primes::Z ) ),
            FS::MaskedAdd( l1, l, int32v( Primes::W ) ) ), x1, y1, z1, w1 );
        float32v n2 = GetGradientDotFancy( HashPrimes( seed,
            FS::MaskedAdd( i2, i, int32v( Primes::X ) ),
            FS::MaskedAdd( j2, j, int32v( Primes::Y ) ),
            FS::MaskedAdd( k2, k, int32v( Primes::Z ) ),
            FS::MaskedAdd( l2, l, int32v( Primes::W ) ) ), x2, y2, z2, w2 );
        float32v n3 = GetGradientDotFancy( HashPrimes( seed,
            FS::MaskedAdd( i3, i, int32v( Primes::X ) ),
            FS::MaskedAdd( j3, j, int32v( Primes::Y ) ),
            FS::MaskedAdd( k3, k, int32v( Primes::Z ) ),
            FS::MaskedAdd( l3, l, int32v( Primes::W ) ) ), x3, y3, z3, w3 );
        float32v n4 = GetGradientDotFancy( HashPrimes( seed, i + int32v( Primes::X ), j + int32v( Primes::Y ), k + int32v( Primes::Z ), l + int32v( Primes::W ) ), x4, y4, z4, w4 );

        return float32v( 33.653125584827855f ) * FS::FMulAdd( n0, t0, FS::FMulAdd( n1, t1, FS::FMulAdd( n2, t2, FS::FMulAdd( n3, t3, n4 * t4 ) ) ) );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::SimplexSmooth, SIMD> final : public virtual FastNoise::SimplexSmooth, public FastSIMD::DispatchClass<FastNoise::ScalableGenerator, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        this->ScalePositions( x, y );

        const float SQRT3 = 1.7320508075688772935274463415059f;
        const float F2 = 0.5f * ( SQRT3 - 1.0f );
        const float G2 = ( SQRT3 - 3.0f ) / 6.0f;

        float32v s = float32v( F2 ) * ( x + y );
        float32v xs = x + s;
        float32v ys = y + s;
        float32v xsb = FS::Floor( xs );
        float32v ysb = FS::Floor( ys );
        float32v xsi = xs - xsb;
        float32v ysi = ys - ysb;
        int32v xsbp = FS::Convert<int32_t>( xsb ) * int32v( Primes::X );
        int32v ysbp = FS::Convert<int32_t>( ysb ) * int32v( Primes::Y );

        mask32v forwardXY = xsi + ysi > float32v( 1.0f );
        float32v boundaryXY = FS::Masked( forwardXY, float32v( -1.0f ) );
        mask32v forwardX = FS::FMulAdd( xsi, float32v( -2.0f ), ysi ) < boundaryXY;
        mask32v forwardY = FS::FMulAdd( ysi, float32v( -2.0f ), xsi ) < boundaryXY;

        float32v t = float32v( G2 ) * ( xsi + ysi );
        float32v xi = xsi + t;
        float32v yi = ysi + t;

        int32v h0 = HashPrimes( seed, xsbp, ysbp );
        float32v v0 = GetGradientDotFancy( h0, xi, yi );
        float32v a = FS::FNMulAdd( xi, xi, FS::FNMulAdd( yi, yi, float32v( 2.0f / 3.0f ) ) );
        float32v a0 = a; a0 *= a0; a0 *= a0;
        float32v value = a0 * v0;

        int32v h1 = HashPrimes( seed, xsbp + int32v( Primes::X ), ysbp + int32v( Primes::Y ) );
        float32v v1 = GetGradientDotFancy( h1, xi - float32v( 2 * G2 + 1 ), yi - float32v( 2 * G2 + 1 ) );
        float32v a1 = FS::FMulAdd( float32v( 2 * ( 1 + 2 * G2 ) * ( 1 / G2 + 2 ) ), t, a + float32v( -2 * ( 1 + 2 * G2 ) * ( 1 + 2 * G2 ) ) );
        a1 *= a1; a1 *= a1;
        value = FS::FMulAdd( a1, v1, value );

        float32v xyDelta = FS::Select( forwardXY, float32v( G2 + 1 ), float32v( -G2 ) );
        xi -= xyDelta;
        yi -= xyDelta;

        int32v h2 = HashPrimes( seed,
            FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardX, xsbp, int32v( Primes::X * 2 ) ), int32v( Primes::X ) ),
            FS::MaskedAdd( forwardXY, ysbp, int32v( Primes::Y ) ) );
        float32v xi2 = xi - FS::Select( forwardX, float32v( 1 + 2 * G2 ), float32v( -1 ) );
        float32v yi2 = FS::MaskedSub( forwardX, yi, float32v( 2 * G2 ) );
        float32v v2 = GetGradientDotFancy( h2, xi2, yi2 );
        float32v a2 = FS::Max( FS::FNMulAdd( xi2, xi2, FS::FNMulAdd( yi2, yi2, float32v( 2.0f / 3.0f ) ) ), float32v( 0 ) );
        a2 *= a2; a2 *= a2;
        value = FS::FMulAdd( a2, v2, value );

        int32v h3 = HashPrimes( seed,
            FS::MaskedAdd( forwardXY, xsbp, int32v( Primes::X ) ),
            FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardY, ysbp, int32v( (int32_t)( Primes::Y * 2LL ) ) ), int32v( Primes::Y ) ) );
        float32v xi3 = FS::MaskedSub( forwardY, xi, float32v( 2 * G2 ) );
        float32v yi3 = yi - FS::Select( forwardY, float32v( 1 + 2 * G2 ), float32v( -1 ) );
        float32v v3 = GetGradientDotFancy( h3, xi3, yi3 );
        float32v a3 = FS::Max( FS::FNMulAdd( xi3, xi3, FS::FNMulAdd( yi3, yi3, float32v( 2.0f / 3.0f ) ) ), float32v( 0 ) );
        a3 *= a3; a3 *= a3;
        value = FS::FMulAdd( a3, v3, value );

        return float32v( 9.28993664146183f ) * value;
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const final
    {
        this->ScalePositions( x, y, z );

        const float F3 = 1.0f / 3.0f;
        const float G3 = -1.0f / 2.0f;

        float32v skewDelta = float32v( F3 ) * ( x + y + z );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;

        // From unit cell base, find closest vertex
        {
            // Perform a double unskew to get the vector whose dot product with skewed vectors produces the unskewed result.
            float32v twiceUnskewDelta = float32v( -0.25f ) * ( dxSkewed + dySkewed + dzSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v xyzNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal

            // Using those, compare scores to determine which vertex is closest.
            constexpr auto considerVertex = [] ( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr
            {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.375f );
            int32v moveMaskBits = FS::Masked( xyzNormal > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, xyzNormal );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b100 );
            maxScore += float32v( 0.125f ) - xyzNormal;
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b110 );

            mask32v moveX = ( moveMaskBits & int32v( 0b001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b100 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );

        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed;
        float32v twiceUnskewDelta = float32v( -0.25f ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v xyzNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal

        float32v unskewDelta = float32v( G3 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 3 * G3 ) * skewedCoordinateSum; // dxBase + dyBase + dzBase

        // Vertex <0, 0, 0>
        float32v value, falloffBaseBase;
        {
            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase ), dxBase, dyBase, dzBase );
            falloffBaseBase = FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( 0.75f ) ) ) ) * float32v( 0.5f );
            value = ( falloffBaseBase * falloffBaseBase ) * ( falloffBaseBase * falloffBaseBase ) * gradientRampValue;
        }

        // Vertex <1, 1, 1>
        {
            mask32v signMask = xyzNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset = float32v( 3 * G3 + 1 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimed ), dxBase - offset, dyBase - offset, dzBase - offset );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset, coordinateSum, float32v( ( 3 * G3 + 1 ) * ( 3 * G3 + 1 ) * ( -3 * 0.5f ) ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0>
        {
            mask32v signMask = xyzNormal < zNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * G3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase ), dxBase, dyBase, dzBase - offset0 );
            float32v falloffBase = FS::Min( ( sign ^ dzBase ) - falloffBaseBase - float32v( ( ( 2 * G3 + 1 ) * ( 2 * G3 + 1 ) * -2 - ( 2 * G3 ) * ( 2 * G3 ) ) * 0.5f ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1>
        {
            mask32v signMask = xyzNormal < yNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * G3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed ), dxBase, dyBase - offset0, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dyBase ) - falloffBaseBase - float32v( ( ( 2 * G3 + 1 ) * ( 2 * G3 + 1 ) * -2 - ( 2 * G3 ) * ( 2 * G3 ) ) * 0.5f ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1>
        {
            mask32v signMask = xyzNormal < xNormal;

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * G3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed ), dxBase - offset0, dyBase, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dxBase ) - falloffBaseBase - float32v( ( ( 2 * G3 + 1 ) * ( 2 * G3 + 1 ) * -2 - ( 2 * G3 ) * ( 2 * G3 ) ) * 0.5f ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }
        
        // Vertex <1, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( G3 ) ^ sign; // offset1 = -offset0 because G3 + 1 = -G3

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase ), dxBase + offset0, dyBase - offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G3 ) * ( G3 ) * -2 - ( G3 + 1 ) * ( G3 + 1 ) ) * 0.5f ) ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( G3 ) ^ sign; // offset1 = -offset0 because G3 + 1 = -G3

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase ), dxBase - offset0, dyBase + offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G3 ) * ( G3 ) * -2 - ( G3 + 1 ) * ( G3 + 1 ) ) * 0.5f ) ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1>
        {
            mask32v signMask = zNormal < float32v( 0 );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( G3 ) ^ sign; // offset1 = -offset0 because G3 + 1 = -G3

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed ), dxBase - offset0, dyBase - offset0, dzBase + offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G3 ) * ( G3 ) * -2 - ( G3 + 1 ) * ( G3 + 1 ) ) * 0.5f ) ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        return value * float32v( 144.736422163332608f );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const final
    {
        this->ScalePositions( x, y, z, w );

        const float SQRT5 = 2.236067977499f;
        const float F4 = ( SQRT5 - 1.0f ) / 4.0f;
        const float G4 = ( SQRT5 - 5.0f ) / 20.0f;

        float32v skewDelta = float32v( F4 ) * ( x + y + z + w );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v wSkewed = w + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v wSkewedBase = FS::Floor( wSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;
        float32v dwSkewed = wSkewed - wSkewedBase;

        // From unit cell base, find closest vertex
        {
            // Perform a double unskew to get the vector whose dot product with skewed vectors produces the unskewed result.
            float32v twiceUnskewDelta = float32v( -0.2f ) * ( dxSkewed + dySkewed + dzSkewed + dwSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v wNormal = dwSkewed + twiceUnskewDelta;
            float32v xyzwNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal + wNormal

            // Using those, compare scores to determine which vertex is closest.
            constexpr auto considerVertex = [] ( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr
            {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.6f ) - xyzwNormal;
            int32v moveMaskBits = FS::Masked( float32v( 0.2f ) > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, float32v( 0.2f ) );
            considerVertex( maxScore, moveMaskBits, -wNormal, 0b0111 );
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b1011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b1101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b1110 );
            maxScore += xyzwNormal - float32v( 0.2f );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b0001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0100 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1000 );
            maxScore += float32v( 0.2f ) - xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0011 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0101 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1001 );
            maxScore += xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal + zNormal, 0b0110 );
            maxScore -= wNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b1010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b1100 );
            
            mask32v moveX = ( moveMaskBits & int32v( 0b0001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b0010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b0100 ) ) != int32v( 0 );
            mask32v moveW = ( moveMaskBits & int32v( 0b1000 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );
            wSkewedBase = FS::MaskedIncrement( moveW, wSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
            dwSkewed = FS::MaskedDecrement( moveW, dwSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );
        int32v wPrimedBase = FS::Convert<int32_t>( wSkewedBase ) * int32v( Primes::W );
        
        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed + dwSkewed;
        float32v twiceUnskewDelta = float32v( -0.2f ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v wNormal = dwSkewed + twiceUnskewDelta;
        float32v xyzwNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal + wNormal

        float32v unskewDelta = float32v( G4 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;
        float32v dwBase = dwSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 4 * G4 ) * skewedCoordinateSum; // dxBase + dyBase + dzBase + dwBase

        // Vertex <0, 0, 0, 0>
        float32v value, falloffBaseBase;
        {
            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimedBase ), dxBase, dyBase, dzBase, dwBase );
            falloffBaseBase = FS::FNMulAdd( dwBase, dwBase, FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( 0.8f ) ) ) ) ) * float32v( 0.5f );
            value = ( falloffBaseBase * falloffBaseBase ) * ( falloffBaseBase * falloffBaseBase ) * gradientRampValue;
        }

        // Vertex <1, 1, 1, 1>
        {
            mask32v signMask = xyzwNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset = float32v( 4 * G4 + 1 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimed ), dxBase - offset, dyBase - offset, dzBase - offset, dwBase - offset );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset, coordinateSum, float32v( ( 4 * G4 + 1 ) * ( 4 * G4 + 1 ) * ( -4 * 0.5f ) ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 1, 0>
        {
            mask32v signMask = xyzwNormal < wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 3 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimedBase ), dxBase - offset1, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset1, coordinateSum, float32v( ( ( 3 * G4 + 1 ) * ( 3 * G4 + 1 ) * -3 - ( 3 * G4 ) * ( 3 * G4 ) ) * 0.5f ) ) - ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0, 1>
        {
            mask32v signMask = xyzwNormal < zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimed ), dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset1, coordinateSum, float32v( ( ( 3 * G4 + 1 ) * ( 3 * G4 + 1 ) * -3 - ( 3 * G4 ) * ( 3 * G4 ) ) * 0.5f ) ) - ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1, 1>
        {
            mask32v signMask = xyzwNormal < yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimed ), dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset1, coordinateSum, float32v( ( ( 3 * G4 + 1 ) * ( 3 * G4 + 1 ) * -3 - ( 3 * G4 ) * ( 3 * G4 ) ) * 0.5f ) ) - ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1, 1>
        {
            mask32v signMask = xyzwNormal < xNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimed ), dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset1, coordinateSum, float32v( ( ( 3 * G4 + 1 ) * ( 3 * G4 + 1 ) * -3 - ( 3 * G4 ) * ( 3 * G4 ) ) * 0.5f ) ) - ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v offset1 = float32v( G4 + 1 ) ^ sign;
            float32v offset0 = float32v( G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimedBase ), dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G4 ) * ( G4 ) * -3 - ( G4 + 1 ) * ( G4 + 1 ) ) * 0.5f ) ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0, 0>
        {
            mask32v signMask = xNormal < -yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimedBase ), dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dxBase + dyBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1, 0>
        {
            mask32v signMask = xNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimedBase ), dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dxBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 0, 1>
        {
            mask32v signMask = xNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimed ), dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dxBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( G4 + 1 ) ^ sign;
            float32v offset0 = float32v( G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimedBase ), dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G4 ) * ( G4 ) * -3 - ( G4 + 1 ) * ( G4 + 1 ) ) * 0.5f ) ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1, 0>
        {
            mask32v signMask = yNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimedBase ), dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dyBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0, 1>
        {
            mask32v signMask = yNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimed ), dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dyBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1, 0>
        {
            mask32v signMask = zNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( G4 + 1 ) ^ sign;
            float32v offset0 = float32v( G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimedBase ), dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G4 ) * ( G4 ) * -3 - ( G4 + 1 ) * ( G4 + 1 ) ) * 0.5f ) ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1, 1>
        {
            mask32v signMask = zNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * G4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimed ), dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( 2 * G4 + 1 ) * ( 2 * G4 + 1 ) * -2 - 2 * ( 2 * G4 ) * ( 2 * G4 ) ) * 0.5f ) ) + ( sign ^ ( dzBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 0, 1>
        {
            mask32v signMask = wNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( G4 + 1 ) ^ sign;
            float32v offset0 = float32v( G4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotFancy( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimed ), dxBase - offset0, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( falloffBaseBase + FS::FMulAdd( offset0, coordinateSum, float32v( ( ( G4 ) * ( G4 ) * -3 - ( G4 + 1 ) * ( G4 + 1 ) ) * 0.5f ) ) + ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        return value * float32v( 115.21625311930542f );
    }
};
