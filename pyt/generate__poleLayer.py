import numpy as np
import os, sys
import gmsh


# ========================================================= #
# ===  generate pole parts of the magnet                === #
# ========================================================= #

def generate__poleLayer( lc=0.0, side="+" ):

    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    #  -- [1-1] constants                           --  #
    eps               = 1.e-10
    x_,y_,z_          = 0, 1, 2
    origin            = [ 0.0, 0.0, 0.0 ]
    lineDim, surfDim  = 1, 2
    
    #  -- [1-2] load parameters                     --  #
    import nkUtilities.load__constants as lcn
    cnsFile           = "dat/parameter.conf"
    const             = lcn.load__constants( inpFile=cnsFile )
    radius            = const["radius"]
    height            = const["height"]
    
    #  -- [1-3] load point Data                     --  #
    inpFile           = "dat/onmesh.dat"
    import nkUtilities.load__pointFile as lpf
    pointData         = lpf.load__pointFile( inpFile=inpFile, returnType="point" )

    #  -- [1-4] load connectivities                 --  #
    inpFile           = "dat/mesh.elements"
    with open( inpFile, "r" ) as f:
        connectivities=np.loadtxt( f )
    connectivities    = np.array( connectivities[:,3:], dtype=np.int64 )
    connectivities    = connectivities - 1

    # ------------------------------------------------- #
    # --- [2] check on Arc elements                 --- #
    # ------------------------------------------------- #
    check       = investigate__onArcElements( connectivities=connectivities, \
                                              pointData=pointData, radius=radius )
    onArc_check = check["onArc_check"]
    onArc_index = check["onArc_index"]
    other_index = check["other_index"]
    
    # ------------------------------------------------- #
    # --- [3] define side surface                   --- #
    # ------------------------------------------------- #
    generate__sideSurface( side=side, radius=radius, height=height, \
                           pointData=pointData, connectivities=connectivities, \
                           onArc_check=onArc_check, onArc_index=onArc_index )

    # ------------------------------------------------- #
    # --- [4] define floor & ceiling sector         --- #
    # ------------------------------------------------- #
    import nkGmshRoutines.generate__sector180 as sec
    floor   = sec.generate__sector180( r1=0.0, r2=radius, zoffset=   0.0, defineSurf=True )
    ceiling = sec.generate__sector180( r1=0.0, r2=radius, zoffset=height, defineSurf=True )

    # ------------------------------------------------- #
    # --- [5] define diameter surface               --- #
    # ------------------------------------------------- #
    generate__diameterSurface( pointData=pointData, radius=radius, height=height )

    # ------------------------------------------------- #
    # --- [6] investigate entity numbers            --- #
    # ------------------------------------------------- #
    entityNum  = investigate__entitiesNumber( height=height )

    # ------------------------------------------------- #
    # --- [7] define pole surface                   --- #
    # ------------------------------------------------- #
    pole_surfs = generate__poleSurface( onArc_index=onArc_index, onArc_check=onArc_check, \
                                        other_index=other_index, connectivities=connectivities, \
                                        pointData=pointData, entityNum=entityNum )
    
    # ------------------------------------------------- #
    # --- [8] volume difinition                     --- #
    # ------------------------------------------------- #
    #  -- [8-1] upper parts                         --  #
    upper_loop = [ entityNum["side_upper"], entityNum["onDia_upper"], \
                   entityNum["ceiling1"], entityNum["ceiling2"] ] + pole_surfs
    s_group_up = gmsh.model.occ.addSurfaceLoop( upper_loop )
    upper_vol  = gmsh.model.occ.addVolume( [ s_group_up ] )
    #  -- [8-2] lower parts                         --  #
    lower_loop = [ entityNum["side_lower"], entityNum["onDia_lower"], \
                   entityNum["floor1"], entityNum["floor2"] ] + pole_surfs
    s_group_lw = gmsh.model.occ.addSurfaceLoop( lower_loop )
    lower_vol  = gmsh.model.occ.addVolume( [ s_group_lw ] )

    # ------------------------------------------------- #
    # --- [9] post-process                          --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    ret = [ lower_vol, upper_vol ]
    return( ret )
    

# ========================================================= #
# ===  onArc element Check                              === #
# ========================================================= #
def investigate__onArcElements( connectivities=None, pointData=None, radius=None ):

    eps              = 1.e-8
    x_, y_, z_       = 0, 1, 2
    
    # ------------------------------------------------- #
    # --- [1] search onArc elements                 --- #
    # ------------------------------------------------- #
    onArc_check    = np.zeros( ( connectivities.shape[0], ), dtype=np.int64 )
    for ik,cnc in enumerate(connectivities):
        pt1 = pointData[ cnc[0], : ]
        pt2 = pointData[ cnc[1], : ]
        pt3 = pointData[ cnc[2], : ]
        dr1 = np.sqrt( pt1[x_]**2 + pt1[y_]**2 ) - radius
        dr2 = np.sqrt( pt2[x_]**2 + pt2[y_]**2 ) - radius
        dr3 = np.sqrt( pt3[x_]**2 + pt3[y_]**2 ) - radius
        if   ( ( dr2 > -eps ) and ( dr2 < +eps ) and ( dr3 > -eps ) and ( dr3 < +eps ) ):
            onArc_check[ik] = 1
        elif ( ( dr1 > -eps ) and ( dr1 < +eps ) and ( dr3 > -eps ) and ( dr3 < +eps ) ):
            onArc_check[ik] = 2
        elif ( ( dr1 > -eps ) and ( dr1 < +eps ) and ( dr2 > -eps ) and ( dr2 < +eps ) ):
            onArc_check[ik] = 3
    onArc_index    = ( np.where( onArc_check >  0 ) )[0]
    other_index    = ( np.where( onArc_check == 0 ) )[0]

    ret = { "onArc_index":onArc_index, "other_index":other_index, "onArc_check":onArc_check }
    return( ret )


# ========================================================= #
# ===  generate side surface                            === #
# ========================================================= #
def generate__sideSurface( pointData=None, connectivities=None, radius=None, height=None, \
                           origin   =[0.0,0.0,0.0], side =None, \
                           onArc_check=None, onArc_index =None  ):

    x_, y_, z_       = 0, 1, 2
    lineDim, surfDim = 1, 2

    # ------------------------------------------------- #
    # --- [1] surface for cut ( tool ) on Arc       --- #
    # ------------------------------------------------- #
    tools = []
    for ie,idx in enumerate( onArc_index ):
        #  - inner node - #
        inn_id  = onArc_check[ idx ]-1
        inn_nd  = connectivities[ idx, inn_id  ]
        inn_pt  = pointData[ inn_nd, : ]
        inn_md  = gmsh.model.occ.addPoint( inn_pt[x_], inn_pt[y_], inn_pt[z_] )
        #  - onArc node - #
        arc_ids = list( set( [0,1,2] ) - set( [inn_id] ) )
        arc_nd1 = connectivities[ idx, arc_ids[0] ]
        arc_nd2 = connectivities[ idx, arc_ids[1] ]
        arc_pt1 = pointData[ arc_nd1, : ]
        arc_pt2 = pointData[ arc_nd2, : ]
        #  - sort ascending order   - #
        if ( arc_pt1[y_] > arc_pt2[y_] ):
            arc_pt1, arc_pt2 = arc_pt2, arc_pt1
            arc_nd1, arc_nd2 = arc_nd2, arc_nd1
        #  - define oversized area  - #
        arc_ex1 = inn_pt + ( arc_pt1 - inn_pt ) * 1.3
        arc_ex2 = inn_pt + ( arc_pt2 - inn_pt ) * 1.3
        arc_md1 = gmsh.model.occ.addPoint( arc_ex1[x_], arc_ex1[y_], arc_ex1[z_] )
        arc_md2 = gmsh.model.occ.addPoint( arc_ex2[x_], arc_ex2[y_], arc_ex2[z_] )
        #  - line definition        - #
        l1 = gmsh.model.occ.addLine( arc_md1, arc_md2 )
        l2 = gmsh.model.occ.addLine( arc_md2, inn_md  )
        l3 = gmsh.model.occ.addLine( inn_md , arc_md1 )
        #  - surface definition     - #
        cl = gmsh.model.occ.addCurveLoop( [l1,l2,l3]  )
        sf = gmsh.model.occ.addPlaneSurface( [cl] )
        tools.append( sf )

    # ------------------------------------------------- #
    # --- [2] define side surface to cut            --- #
    # ------------------------------------------------- #
    #  -- [2-1] side "+" case                       --  #
    if ( side == "+" ):
        pth1         = -90.0 / 180.0 * np.pi
        pth2         = +90.0 / 180.0 * np.pi
        arc1         = gmsh.model.occ.addCircle( origin[x_], origin[y_], origin[z_], \
                                                 radius, angle1=pth1, angle2=pth2 )
        dx,dy,dz     = 0.0, 0.0, height
        ret          = gmsh.model.occ.extrude( [(lineDim,arc1)], dx,dy,dz )
        circumf_surf = ret[1][1]

    # ------------------------------------------------- #
    # --- [3] boolean cut of the surfaces           --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    target = [(surfDim,circumf_surf)]
    tools  = [(surfDim,int( tool ) ) for tool in tools ]
    ret    = gmsh.model.occ.cut( target, tools, removeObject=False, removeTool=False )
    gmsh.model.occ.remove( tools, recursive=True )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    return()





# ========================================================= #
# ===  define diameter surface                          === #
# ========================================================= #
def generate__diameterSurface( pointData=None, radius=None, height=None ):

    eps              = 1.e-8
    x_, y_, z_       = 0, 1, 2
    lineDim, surfDim = 1, 2

    # ------------------------------------------------- #
    # --- [1] collect on Diameter points            --- #
    # ------------------------------------------------- #
    index            = np.where( ( pointData[:,x_] >= - eps ) & \
                                 ( pointData[:,x_] <= + eps ) )
    onDia_points     = pointData[index]
    index            = np.argsort( onDia_points[:,y_] )
    onDia_points     = onDia_points[index][:]

    # ------------------------------------------------- #
    # --- [2] points to be plot ( Quad )            --- #
    # ------------------------------------------------- #
    square_width     = 0.1 * radius
    onDia_ext1       = np.copy( onDia_points )
    onDia_ext2       = np.copy( onDia_points )
    onDia_ext1[:,x_] = - square_width
    onDia_ext2[:,x_] = + square_width
    nDiameter        = onDia_points.shape[0]

    # ------------------------------------------------- #
    # --- [3]  add points to the model              --- #
    # ------------------------------------------------- #
    pnums1, pnums2 = [], []
    for ik in range( nDiameter ):
        pt1   = onDia_ext1[ik]
        pt2   = onDia_ext2[ik]
        pnum1 = gmsh.model.occ.addPoint( pt1[x_], pt1[y_], pt1[z_] )
        pnum2 = gmsh.model.occ.addPoint( pt2[x_], pt2[y_], pt2[z_] )
        pnums1.append( pnum1 )
        pnums2.append( pnum2 )

    # ------------------------------------------------- #
    # --- [4] define quad to cut                    --- #
    # ------------------------------------------------- #
    tools = []
    for ik in range( nDiameter-1 ):
        l1 = gmsh.model.occ.addLine( pnums1[ik  ], pnums1[ik+1] )
        l2 = gmsh.model.occ.addLine( pnums1[ik+1], pnums2[ik+1] )
        l3 = gmsh.model.occ.addLine( pnums2[ik+1], pnums2[ik  ] )
        l4 = gmsh.model.occ.addLine( pnums2[ik  ], pnums1[ik  ] )
        cl = gmsh.model.occ.addCurveLoop( [l1,l2,l3,l4] )
        sf = gmsh.model.occ.addPlaneSurface( [cl] )
        tools.append( sf )
        
    # ------------------------------------------------- #
    # --- [5] define diameter surface to be cut     --- #
    # ------------------------------------------------- #
    import nkGmshRoutines.generate__quadShape as gqs
    x1,x2,x3,x4  = [ [ 0.0, - radius,    0.0 ],
                     [ 0.0, + radius,    0.0 ],
                     [ 0.0, + radius, height ],
                     [ 0.0, - radius, height ] ]
    ret          = gqs.generate__quadShape( x1=x1, x2=x2, x3=x3, x4=x4 )
    onDia_surf   = ret["surf"]["quad"]
    
    # ------------------------------------------------- #
    # --- [6] boolean cut of the surface            --- #
    # ------------------------------------------------- #
    target = [(surfDim,onDia_surf)]
    tools  = [(surfDim,int( tool ) ) for tool in tools ]
    gmsh.model.occ.synchronize()
    ret    = gmsh.model.occ.cut( target, tools, removeObject=False, removeTool=False )
    gmsh.model.occ.remove( tools, recursive=True )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()


# ========================================================= #
# ===  obtain elemental entity number                   === #
# ========================================================= #

def investigate__entitiesNumber( height=None ):

    eps              = 1.e-8
    x_, y_, z_       = 0, 1, 2
    lineDim, surfDim = 1, 2
    
    # ------------------------------------------------- #
    # --- [1] obtain elemental entity number        --- #
    # ------------------------------------------------- #
    #  -- [1-1] grouping                            --  #
    s_dimtag                 = gmsh.model.getEntities(2)
    surfs                    = [ int( dimtag[1] ) for dimtag in s_dimtag ]
    ceilings, floors, onDia = [], [], []
    for iS,surfnum in enumerate(surfs):
        CoM = gmsh.model.occ.getCenterOfMass( surfDim, surfnum )
        if   ( ( CoM[x_]          > -eps ) and ( CoM[x_]          < +eps ) ):
            onDia   .append( surfnum )
        elif ( ( CoM[z_]          > -eps ) and ( CoM[z_]          < +eps ) ):
            floors  .append( surfnum )
        elif ( ( CoM[z_] - height > -eps ) and ( CoM[z_] - height < +eps ) ):
            ceilings.append( surfnum )
    side_surfs = list( set( surfs ) - set( ceilings ) - set( floors ) - set( onDia ) )
    #  -- [1-2] labeling                             -- #
    z1 = ( gmsh.model.occ.getCenterOfMass( surfDim, side_surfs[0] ) )[z_]
    z2 = ( gmsh.model.occ.getCenterOfMass( surfDim, side_surfs[1] ) )[z_]
    z3 = ( gmsh.model.occ.getCenterOfMass( surfDim, onDia[0] )     )[z_]
    z4 = ( gmsh.model.occ.getCenterOfMass( surfDim, onDia[1] )     )[z_]
    if ( z1 > z2 ):
        side_upper, side_lower = side_surfs[0], side_surfs[1]
    else:
        side_upper, side_lower = side_surfs[1], side_surfs[0]
    if ( z3 > z4 ):
        onDia_upper, onDia_lower = onDia[0], onDia[1]
    else:
        onDia_upper, onDia_lower = onDia[1], onDia[0]
    print( "floors       :: {0}".format( floors       ) )
    print( "ceilings     :: {0}".format( ceilings     ) )
    print( "side_upper   :: {0}".format( side_upper   ) )
    print( "side_lower   :: {0}".format( side_lower   ) )
    print( "onDia_upper :: {0}".format( onDia_upper ) )
    print( "onDia_lower :: {0}".format( onDia_lower ) )

    ret = {}
    ret["floor1"]     , ret["floor2"]      =   floors[0],   floors[1]
    ret["ceiling1"]   , ret["ceiling2"]    = ceilings[0], ceilings[1]
    ret["side_upper"] , ret["side_lower"]  =  side_upper,  side_lower
    ret["onDia_upper"], ret["onDia_lower"] = onDia_upper, onDia_lower
    return( ret )
    

# ========================================================= #
# ===  define pole surfaces                             === #
# ========================================================= #

def generate__poleSurface( onArc_index=None, onArc_check=None, other_index=None, \
                           connectivities=None, pointData=None, entityNum=None ):

    eps              = 1.e-8
    x_, y_, z_       = 0, 1, 2
    lineDim, surfDim = 1, 2

    # ------------------------------------------------- #
    # --- [1] investigate line number of division   --- #
    # ------------------------------------------------- #
    side_upper       = [(surfDim,entityNum["side_upper"])]
    side_lower       = [(surfDim,entityNum["side_lower"])]
    arc_pieces_upper = gmsh.model.getBoundary( side_upper, oriented=False )
    arc_pieces_lower = gmsh.model.getBoundary( side_lower, oriented=False )
    arc_pieces_upper = set( [ int( dimtag[1] ) for dimtag in arc_pieces_upper ] )
    arc_pieces_lower = set( [ int( dimtag[1] ) for dimtag in arc_pieces_lower ] )
    arc_pieces       = list( arc_pieces_upper & arc_pieces_lower )

    # ------------------------------------------------- #
    # --- [2] label dividing line by end-point      --- #
    # ------------------------------------------------- #
    ptkeys           = []
    arc_lines        = {}
    for iL,piece in enumerate( arc_pieces ):
        ret     = gmsh.model.getBoundary( [(lineDim,piece)] )
        pt1,pt2 = int( ret[0][1] ), int( ret[1][1] )
        if ( pt1 > pt2 ):
            pt1, pt2 = pt2, pt1
        ptkey            = "{0}_{1}".format( pt1, pt2 )
        arc_lines[ptkey] = piece
        ptkeys.append( ptkey )

    # radii          = np.sqrt( pointData[:,x_]**2 + pointData[:,y_]**2 )
    # index          = np.where( ( radius-eps < radii ) & \
    #                            ( radius+eps > radii ) )
    # nodeNum        = ( np.arange( 1, pointData.shape[0]+1 ) )[index]
    # onDia_points   = pointData[index]

    
    # ------------------------------------------------- #
    # --- [3] define onArc surfaces                 --- #
    # ------------------------------------------------- #
    arc_surfs   = []
    for ie,idx in enumerate( onArc_index ):
        #  - inner node - #
        inn_id  = onArc_check[ idx ]-1
        inn_nd  = connectivities[ idx, inn_id  ]
        inn_pt  = pointData[ inn_nd, : ]
        inn_md  = gmsh.model.occ.addPoint( inn_pt[x_], inn_pt[y_], inn_pt[z_] )
        #  - onArc node - #
        ids = list( set( [0,1,2] ) - set( [inn_id] ) )
        nd1 = connectivities[ idx, ids[0] ]
        nd2 = connectivities[ idx, ids[1] ]
        pt1 = pointData[ nd1, : ]
        pt2 = pointData[ nd2, : ]
        md1 = gmsh.model.getEntitiesInBoundingBox( pt1[x_]-eps, pt1[y_]-eps, pt1[z_]-eps, \
                                                   pt1[x_]+eps, pt1[y_]+eps, pt1[z_]+eps, dim=0 )
        md2 = gmsh.model.getEntitiesInBoundingBox( pt2[x_]-eps, pt2[y_]-eps, pt2[z_]-eps, \
                                                   pt2[x_]+eps, pt2[y_]+eps, pt2[z_]+eps, dim=0 )
        md1,md2    = md1[0][1], md2[0][1]
        key        = "{0}_{1}".format( min( md1, md2 ), max( md1, md2 ) )
        line12     = gmsh.model.occ.addLine( inn_md,    md1 )
        line31     = gmsh.model.occ.addLine(    md2, inn_md )
        arc23      = arc_lines[key]
        
        l_group    = gmsh.model.occ.addCurveLoop( [ line12, arc23, line31 ] )
        s_num      = gmsh.model.occ.addPlaneSurface( [ l_group ] )
        arc_surfs.append( s_num )

    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
        
    # ------------------------------------------------- #
    # --- [4] define normal triangles               --- #
    # ------------------------------------------------- #
    tris = []
    for ie,idx in enumerate( other_index ):
        mds, lns = [], []
        for iv in [0,1,2]:
            nd = connectivities[idx,iv]
            pt = pointData[ nd, : ]
            md = gmsh.model.occ.addPoint( pt[x_], pt[y_], pt[z_] )
            mds.append( md )
        for iv in [ [0,1], [1,2], [2,0] ]:
            ln = gmsh.model.occ.addLine( mds[ iv[0] ], mds[ iv[1] ] )
            lns.append( ln )
        l_Group = gmsh.model.occ.addCurveLoop( [ lns[0], lns[1], lns[2] ] )
        tri     = gmsh.model.occ.addPlaneSurface( [l_Group] )
        tris.append( tri )

    # ------------------------------------------------- #
    # --- [5] post-process                          --- #
    # ------------------------------------------------- #
    
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    ret = tris + arc_surfs
    return( ret )
        



# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #
if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] initialization of the gmsh            --- #
    # ------------------------------------------------- #
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal", 1 )
    gmsh.model.add( "model" )
    
    # ------------------------------------------------- #
    # --- [2] define model                          --- #
    # ------------------------------------------------- #
    generate__poleLayer()
    
    # ------------------------------------------------- #
    # --- [3] post process                          --- #
    # ------------------------------------------------- #
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", 0.05 )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", 0.05 )
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.generate(3)
    gmsh.write( "msh/model.msh" )
    gmsh.finalize()







