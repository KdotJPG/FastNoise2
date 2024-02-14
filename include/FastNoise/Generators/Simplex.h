#pragma once
#include "Generator.h"

namespace FastNoise
{
    class Simplex : public virtual ScalableGenerator
    {
    public:
        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<Simplex> : MetadataT<ScalableGenerator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Coherent Noise" );

            description = 
                "Smooth gradient noise from an N dimensional simplex grid\n"
                "Developed by Ken Perlin in 2001";
        }
    };
#endif

    class SimplexSmooth : public virtual Generator
    {
    public:        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<SimplexSmooth> : MetadataT<ScalableGenerator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Coherent Noise" );

            description =
                "Smoother gradient noise from an N dimensional simplex grid\n"
                "Developed by K.jpg in 2017";
        }
    };
#endif
}
