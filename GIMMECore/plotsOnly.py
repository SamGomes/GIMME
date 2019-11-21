

import matplotlib.pyplot as plt


from numpy import array

maxNumTrainingIterations = 50
numRealIterations = 10


GIMMEAbilities = [0.0, 0.00028207740852289476, 0.0004565651966564164, 0.0005083362104390593, 0.0005692786761471083, 0.0005774582508380157, 0.0005879711045648562, 0.0005947699934158529, 0.0005812049729686404, 0.0006461842745834419, 0.0005847648961663214, 0.0005867519694579811, 0.0006144414053426146, 0.0006044595640853516, 0.0006033771649770592, 0.0006246608053044673, 0.0006228580581881999, 0.0006145813459481258, 0.0006250901989603944, 0.000575076071326693, 0.0005691021112865917, 0.0006010659694092251, 0.0006086258845428812, 0.0005856023143651142, 0.0006185580329597198, 0.0006032573973273216, 0.0006366513911337243, 0.0006044435801935841, 0.0006110381224403797, 0.0006317235341441252, 0.0006074781307602584, 0.0006223330545779916, 0.0006030730118551848, 0.0006377739065342688, 0.0005999511021435801, 0.0005969168877049762, 0.0006211710419160166, 0.0006096749142929911, 0.0006268323639350841, 0.0006050878968468895, 0.0006022738354060029, 0.0005944564232777052, 0.0005899511367139306, 0.0006017157182128672, 0.0006007040396615942, 0.00061097037855718, 0.0006243898456773378, 0.0006307549784020809, 0.0006023182542704945, 0.0006099039281686636, 0.0006013930355325386, 0.000622342618421411, 0.0006155006452992645, 0.0005880682797593297, 0.0005984542269192136, 0.0005729320392023911, 0.0006215219035016145, 0.0006062248585809444, 0.0006351224422108364, 0.0006187048021639312, 0.0005914959948230168]
GIMMEEngagements = [0.0, 0.0006983157871229883, 0.0010913959228536624, 0.0012878110225347272, 0.0013910044434905515, 0.0014517888575040947, 0.0014801904914140346, 0.001489686714289908, 0.001499045773938281, 0.0015088232529520003, 0.0015116511892219113, 0.0015106122995362113, 0.0015131102277031955, 0.0015115144662342277, 0.0015159654444366483, 0.0015176431179642557, 0.0015142840118137523, 0.0015147908158987685, 0.0015146438607566604, 0.0015184985875767913, 0.001519077498349491, 0.0015186651451035943, 0.0015101845186737352, 0.0015123110425709135, 0.0015162071042356437, 0.0015146219200786676, 0.0015179753963487064, 0.0015113799611439786, 0.0015096709988933696, 0.0015117862309011326, 0.0015114205395978875, 0.0015158090408326872, 0.0015153595722475716, 0.0015183833695679096, 0.0015166730734446885, 0.0015094197846749256, 0.0015089971959890542, 0.0015163508698370523, 0.0015130346868924038, 0.0015121582002401463, 0.0015229024529462855, 0.0015195283941232987, 0.0015155886344324178, 0.0015116232291627946, 0.001513993969228539, 0.0015174130168601253, 0.0015203743699251427, 0.001522132306301956, 0.001521656295221831, 0.0015167310408658054, 0.0015180058281099504, 0.001513820398530264, 0.0015138354214158422, 0.001514252576684943, 0.0015130680939863124, 0.0015129125427004663, 0.0015088678235667162, 0.001507937062804222, 0.0015109215699309062, 0.001509779695126973, 0.0015123728217388266]
GIMMEPrefProfDiff = [0.0, 0.003071614327676452, 0.0028440031651995427, 0.0028456296459415205, 0.0028142595461956143, 0.002785173041899284, 0.0027910143801965167, 0.002798392448772512, 0.0027957086112969016, 0.002774087805777502, 0.002769236943235177, 0.002779803222584899, 0.0027677194516247945, 0.0027894496262563895, 0.002767625849511115, 0.0027790901684835788, 0.002773998905611494, 0.002775919663936468, 0.0027807865293942497, 0.002770514200029322, 0.0027738833817395996, 0.0027716458603450783, 0.0027949063988129187, 0.002776194035773269, 0.002782030544253825, 0.002772664183586251, 0.0027460660133728997, 0.0027807743529837847, 0.002807185252009952, 0.00276843308626489, 0.002777663792963344, 0.0027548716448684133, 0.002777764305568948, 0.0027481143439860014, 0.0027596989826786955, 0.002801343093874741, 0.002783282391261616, 0.0027480791020711597, 0.0027810375197813942, 0.0027626553722176752, 0.0027314233249310814, 0.002780470476059458, 0.0027829002757254646, 0.0027760582006535212, 0.002761285557428939, 0.002747994801648277, 0.0027511166065043767, 0.002760564757379181, 0.002744218040575161, 0.002767695930428951, 0.0027635141760411638, 0.00279451563139823, 0.0027809096135449386, 0.002778078321977084, 0.00276744038510239, 0.002793030109502696, 0.0027848929215996306, 0.002791652842263597, 0.0027619973139996375, 0.002790354011863016, 0.002766467024034131]

randomAbilities = [0.0, 0.00027810441271505386, 0.00043314876877454924, 0.0004798823053407875, 0.0005313292063337266, 0.0005323535033820486, 0.000539073843495273, 0.0005431416174602797, 0.0005311207573226945, 0.0005861204275590576, 0.0005327687492854171, 0.0005337412645531844, 0.0005614294653593724, 0.0005568364464710135, 0.0005557181037215034, 0.0005697712740726729, 0.0005719459525046619, 0.0005612328555006181, 0.0005736378587952436, 0.0005255873186099447, 0.0005202493877889828, 0.0005466476755256176, 0.0005565978361318645, 0.000536725109933342, 0.000561475733666065, 0.000549599338540706, 0.0005786961101814242, 0.0005532938623474246, 0.000560061795096198, 0.000578397492648587, 0.0005537728187509708, 0.000566260781290358, 0.0005490023291078933, 0.0005798156773546539, 0.0005517680882466062, 0.0005475078567978218, 0.0005691759590065402, 0.000558461984281143, 0.0005784358644650145, 0.0005571016341128056, 0.0005491937951576153, 0.0005449316623650829, 0.0005421512743314796, 0.0005553524048302644, 0.0005514812101882443, 0.000556035842764415, 0.0005656868824632817, 0.0005696307779691576, 0.0005497578723941926, 0.0005603959836519662, 0.000548896496909844, 0.0005674592784298482, 0.0005613502690216938, 0.000536340268667144, 0.0005490914437422392, 0.0005277377772716027, 0.0005728128276624772, 0.0005597415286641345, 0.0005856421334946867, 0.0005695392502784703, 0.0005434195305691317]
randomEngagements = [0.0, 0.0006893354169320322, 0.0010359996989288266, 0.0012181401388581506, 0.0012985437699914563, 0.0013380691979050273, 0.0013581976433612321, 0.001362448849052182, 0.0013702938281669047, 0.0013704122642172983, 0.0013773653045003292, 0.0013744346762121632, 0.0013841998949585305, 0.001389671420036177, 0.0013964576939507316, 0.001386415364501217, 0.0013915511939333982, 0.0013836409567936513, 0.0013884274622978352, 0.0013870774416949948, 0.001390053342473961, 0.001381546092249218, 0.0013814348796255844, 0.0013858067755924337, 0.0013772428260840235, 0.0013803320286381148, 0.0013794211634728377, 0.0013847219888522954, 0.001384759199372305, 0.001384043488050436, 0.0013783983091728888, 0.0013793164000997834, 0.001377010911402163, 0.001379050470721862, 0.0013941208941609259, 0.0013844206319605274, 0.0013828872646571098, 0.0013866730059037943, 0.001394724291740807, 0.0013934412864398881, 0.0013871504631376536, 0.0013899393593535243, 0.0013873998166988226, 0.0013959619540166884, 0.0013888096464351701, 0.001382584128657081, 0.0013768999345154754, 0.0013764386351097085, 0.0013873169172104668, 0.0013920465575279848, 0.0013852024060033205, 0.0013799510450075278, 0.0013803017440238551, 0.0013800348713492157, 0.0013886927166504599, 0.0013959909101951792, 0.0013886064129196257, 0.001395490240052889, 0.0013948006268087942, 0.001389169395968413, 0.001390827644613647]
randomPrefProfDiff = [0.0, 0.0030807610315441263, 0.0030788633837068088, 0.003044450232735065, 0.003080102708028138, 0.003092610788719148, 0.003085588587141791, 0.003112660367351811, 0.00308698158469519, 0.0030982758507415004, 0.0030531971556559223, 0.0030992552191750136, 0.0030489926051696127, 0.003026286076757682, 0.003033444653393244, 0.00307928537843515, 0.0030414696231667046, 0.0031032618570070378, 0.0030545401418088924, 0.0030645401427974576, 0.0030494860633281363, 0.0030923949435585794, 0.003077018899853635, 0.0030613680554540624, 0.003111976974678561, 0.0030748324064695694, 0.003091327893201363, 0.0030661180347645784, 0.0030547298051449476, 0.003074503943825115, 0.0030980445223002854, 0.0030622282526814055, 0.003085138613171381, 0.0030803711191811894, 0.0030286934898405924, 0.003088520121503083, 0.003085432121591058, 0.0030539629000781118, 0.0030306139311863265, 0.0030566421588138367, 0.0030685725187998354, 0.0030479483942343314, 0.0030841054980848335, 0.0030285806876718035, 0.003082250013472414, 0.0030879135799303952, 0.003088126355797801, 0.0030997183861039977, 0.0030486618870485807, 0.0030496744469352827, 0.003080927264850994, 0.0030836080104922584, 0.0030972934749829584, 0.003077463273707533, 0.003059621867087344, 0.0030287658449776387, 0.0030847501499014942, 0.0030496465103893497, 0.0030458067491094154, 0.003083894154785512, 0.003063159149055977]

optimalAbilities = [0.0, 0.0003226894385483616, 0.0005019814999833501, 0.0005520998751654482, 0.0006145036052795041, 0.0006147942011184793, 0.0006240229496561428, 0.0006325250581830929, 0.0006166494694191454, 0.0006847727867260148, 0.0006194663046383292, 0.0006226554378940953, 0.0006490939451463533, 0.0006396986068193877, 0.0006374116549166832, 0.0006580343854184436, 0.0006568498226040232, 0.0006480043985502645, 0.0006584868164021212, 0.0006071053311310866, 0.000598927335254067, 0.0006307335322598476, 0.000641752037428106, 0.0006173582611239465, 0.000651642316972603, 0.0006366067351033746, 0.0006722644761579626, 0.0006378190980569012, 0.0006460341077601082, 0.0006665489811346918, 0.0006407368543820049, 0.0006563822133473866, 0.0006338239859948252, 0.000672479226063223, 0.0006331164876939926, 0.0006310868496617403, 0.0006597324590491697, 0.0006441484956368602, 0.0006626132351958563, 0.0006408773010385133, 0.000631741248765101, 0.0006261118866146907, 0.0006226714909835475, 0.0006352535029269772, 0.0006347296591924048, 0.0006436671441339923, 0.0006587459336274939, 0.0006638447919712878, 0.000633875829366352, 0.0006435106439903817, 0.00063416715194166, 0.0006585008938375538, 0.0006499176321106817, 0.0006208904413925257, 0.0006316181091320717, 0.0006038054793735781, 0.0006583350938685951, 0.0006423997729833264, 0.0006728706912104247, 0.0006535730773261589, 0.0006255753617403211]
optimalEngagements = [0.0, 0.000793589169762058, 0.0011940708837584247, 0.0013933681849623214, 0.001491667654179999, 0.0015383593968792601, 0.0015606173811067874, 0.001569779168086826, 0.0015766597202626154, 0.00158595967661269, 0.0015922052973678896, 0.0015898064334155706, 0.0015907800330764558, 0.0015891742176993913, 0.0015937583131796994, 0.0015908754107087078, 0.0015906650480074687, 0.001588408366159835, 0.001586039061051026, 0.0015872592051782248, 0.0015846316100956406, 0.0015852435205810508, 0.0015844871215363314, 0.0015869349182165292, 0.0015917189700504915, 0.0015907990922232416, 0.0015922770159399104, 0.001585402180412881, 0.0015867785112507334, 0.001587780338662377, 0.0015868227218287759, 0.0015904055449506742, 0.001585357827487472, 0.001588755321606993, 0.0015838988330876722, 0.001587375099909769, 0.0015940000209667235, 0.001591988982477108, 0.0015925857889176435, 0.0015919826267713107, 0.001587896911986989, 0.0015904076590238531, 0.0015862004263155237, 0.001588565160769934, 0.001590313649314613, 0.0015936287760476456, 0.0015937962507275778, 0.0015948594426855732, 0.0015921561805005635, 0.0015929268836547218, 0.0015925204875847706, 0.0015941636944028573, 0.0015917629739609098, 0.0015882725771939087, 0.0015865717083245288, 0.0015805495752838196, 0.0015893935773458863, 0.0015899304926057527, 0.0015867358710295151, 0.0015865443890278224, 0.0015881884441137427]
optimalPrefProfDiff = [0.0, 0.002565386797937977, 0.002542692584355038, 0.0025477304823141715, 0.002550544669287148, 0.0025449024436534295, 0.0025677022850361265, 0.0025617746781083153, 0.0025509334208789563, 0.0025516725143289983, 0.002542471614110998, 0.0025662711705195826, 0.0025514286865762816, 0.00256575941120576, 0.002526582222364376, 0.002550334948360306, 0.0025500862610946173, 0.0025627274337545534, 0.0025515053282664226, 0.002557443702396155, 0.0025529652541118747, 0.002554277056484739, 0.002562405189832652, 0.002547161018135783, 0.002547075679576163, 0.0025535073976903676, 0.002549036872384322, 0.0025615474646872058, 0.0025740565438768254, 0.002553512213447792, 0.0025624879738669745, 0.002556593384787893, 0.0025803864582019873, 0.0025571525735520557, 0.002576155358934238, 0.002546257695397748, 0.002527959156081062, 0.002555447981807633, 0.0025531616320393197, 0.002564634934481687, 0.002568827636537449, 0.002556814549098657, 0.0025693093703422487, 0.0025502895726266413, 0.0025568676562770193, 0.0025355876961366125, 0.0025406078253916104, 0.0025460960499569148, 0.0025605601430192058, 0.0025431398014419617, 0.0025615358648662517, 0.002532960147543444, 0.002548398877604506, 0.0025610300376130947, 0.0025551679719526313, 0.002568233872035173, 0.0025448413469779754, 0.002548362246700193, 0.002568361520052474, 0.0025520135624718734, 0.0025455525332619628]

timesteps=[i for i in range(maxNumTrainingIterations + numRealIterations + 1)]
# -------------------------------------------------------
# plt.plot(timesteps, GIMME10Abilities, label=r'$GIMME 10 Samples (avg. it. exec. time = '+str(GIMME10ExecTime)+')$')
# plt.plot(timesteps, GIMMEAbilities, label=r'$GIMME 100 Samples (avg. it. exec. time = '+str(GIMMEExecTime)+')$')
# plt.plot(timesteps, GIMME1000Abilities, label=r'$GIMME 1000 Samples (avg. it. exec. time = '+str(GIMME1000ExecTime)+')$')
# plt.plot(timesteps, GIMME2000Abilities, label=r'$GIMME 2000 Samples (avg. it. exec. time = '+str(GIMME2000ExecTime)+')$')

# plt.xlabel("Iteration")
# plt.ylabel("avg Ability Increase")

# plt.savefig(newpath+'/charts/simulationsResultsNSamplesComparison.png')

# plt.legend(loc='best')
# plt.show()

# # -------------------------------------------------------
# plt.plot(timesteps, GIMMEK1Abilities, label=r'$GIMME k = 1$')
# plt.plot(timesteps, GIMMEAbilities, label=r'$GIMME k = 5$')
# plt.plot(timesteps, GIMMEK24Abilities, label=r'$GIMME k = 24$')
# plt.plot(timesteps, GIMMEK30Abilities, label=r'$GIMME k = 30$')

# plt.xlabel("Iteration")
# plt.ylabel("avg Ability Increase")

# plt.savefig(newpath+'/charts/simulationsResultsNSamplesAndNNsComparison.png')

# plt.legend(loc='best')
# plt.show()


# -------------------------------------------------------
plt.plot(timesteps, GIMMEAbilities, label=r'$GIMME\ strategy$')
plt.plot(timesteps, randomAbilities, label=r'$Random\ strategy$')
plt.plot(timesteps, optimalAbilities, label=r'$Optimal\ strategy$')

plt.xlabel("Iteration")
plt.ylabel("avg Ability Increase")

# plt.savefig(newpath+'/charts/simulationsResultsAbility_old.png')

plt.legend(loc='best')
plt.show()


# -------------------------------------------------------
plt.plot(timesteps, array(optimalAbilities) - array(GIMMEAbilities), label=r'$GIMME\ strategy$')
plt.plot(timesteps, array(optimalAbilities) - array(randomAbilities), label=r'$Random\ strategy$')

plt.xlabel("Iteration")
plt.ylabel("Distance from Optimal avg Ability Increase")

# plt.savefig(newpath+'/charts/simulationsResultsAbility.png')

plt.legend(loc='best')
plt.show()



# -------------------------------------------------------
plt.plot(timesteps, GIMMEEngagements, label=r'$GIMME\ strategy$')
plt.plot(timesteps, randomEngagements, label=r'$Random\ strategy$')
plt.plot(timesteps, optimalEngagements, label=r'$Optimal\ strategy$')

plt.xlabel("Iteration")
plt.ylabel("Engagement Increase")

# plt.savefig(newpath+'/charts/simulationsResultsEngagement.png')

plt.legend(loc='best')
plt.show()


# -------------------------------------------------------
plt.plot(timesteps, GIMMEPrefProfDiff, label=r'$GIMME\ strategy$')
plt.plot(timesteps, randomPrefProfDiff, label=r'$Random\ strategy$')
plt.plot(timesteps, optimalPrefProfDiff, label=r'$Optimal\ strategy$')

plt.xlabel("Iteration")
plt.ylabel("avg Preference Differences")

# plt.savefig(newpath+'/charts/simulationsResultsProfileDist.png')

plt.legend(loc='best')
plt.show()

