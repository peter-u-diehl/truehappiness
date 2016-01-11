'''
Created on 09.07.2015

@author: peter
'''

import numpy as np
from pylab import *
from gensim.models import Word2Vec

weight_path = './single_layer_hedonometer/' #  'random/' #  
img_dict = {}
img_dict['start'] = imread(weight_path + 'sleep.png')
for number in xrange(1,9):
    img_dict[number] = imread(weight_path + str(number)+'.png')
fig = figure(frameon=False)
ax = fig.add_subplot(111)
ax.xaxis.set_ticklabels([None])
ax.yaxis.set_ticklabels([None])
ax.xaxis.set_ticks([None])
ax.yaxis.set_ticks([None])
axis('off')
# print type(img_dict['start']), img_dict['start'].shape
imshow(img_dict['start'])
ion()
show()

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------

import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from brian import *
  
# import brian.experimental.cuda.gpucodegen as gpu

single_example_time =   1.0 * b.second 
resting_time = 0.0 * b.second
timestep = 0.001

v_reset = 0
v_thresh = 4
refrac = 0. * b.ms
max_input_rate = 1000
n_input = 64

poisson_inputs = True
num_layers = 2
num_neurons = [64, 1]
input_population_names = ['input']
conn_structure = 'dense' # sparse
record_spikes = False
b.ion()

print u"\U0001F637"

b.set_global_preferences( 
                        defaultclock = b.Clock(dt=timestep*b.second), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave=True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimisations are turned on 
                        #- if you need IEEE guaranteed results, turn this switch off.
                        useweave_linear_diffeq = False,  # Whether to use weave C++ acceleration for the solution of linear differential 
                        #equations. Note that on some platforms, typically older ones, this is faster and on some platforms, 
                        #typically new ones, this is actually slower.
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenreset = True,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = True,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = False,  # Whether or not to use OpenMP pragmas in generated C code. 
                        #If supported on your compiler (gcc 4.2+) it will use multiple CPUs and can run substantially faster.
                        magic_useframes = True,  # Defines whether or not the magic functions should search for objects 
                        #defined only in the calling frame or if they should find all objects defined in any frame. 
                        #This should be set to False if you are using Brian from an interactive shell like IDLE or IPython 
                        #where each command has its own frame, otherwise set it to True.
                       ) 



neuron_eqs = '''
        dv/dt = 0  : volt
        '''

        
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
state_monitors = {}

neuron_groups['e'] = b.NeuronGroup(np.sum(num_neurons), neuron_eqs, threshold= v_thresh, refractory= refrac, reset= v_reset, 
                 compile = True, freeze = True)


#------------------------------------------------------------------------------ 
# create network population
#------------------------------------------------------------------------------ 
for i in xrange(num_layers):
#     print 'create neuron group', i
    
    neuron_groups[i] = neuron_groups['e'].subgroup(num_neurons[i])

#     print 'create monitors for', i
    rate_monitors[i] = b.PopulationRateMonitor(neuron_groups[i], bin = (single_example_time+resting_time)/b.second)
    spike_counters[i] = b.SpikeCounter(neuron_groups[i])
    
    if record_spikes:
        spike_monitors[i] = b.SpikeMonitor(neuron_groups[i])

        state_monitors[i] = b.MultiStateMonitor(neuron_groups[i], ['v'], record=[0])

if record_spikes:
    b.figure()
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors[1], refresh=1000*b.ms, showlast=1000*b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors[1], refresh=1000*b.ms, showlast=1000*b.ms)



#------------------------------------------------------------------------------ 
# create input population
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    if poisson_inputs:
        input_groups[name] = b.PoissonGroup(n_input*2, 0)
    else:
        input_groups[name] = b.SpikeGeneratorGroup(n_input*2, [])

    rate_monitors[name] = b.PopulationRateMonitor(input_groups[name], bin = (single_example_time+resting_time)/b.second)


#------------------------------------------------------------------------------ 
# create connections from input populations to network populations
#------------------------------------------------------------------------------ 
weight_matrix = np.zeros((n_input*2, num_neurons[0]))
temp = np.loadtxt(weight_path + '0.txt')
print np.min(temp), np.max(temp)
weight_matrix[:n_input,:] = temp
weight_matrix[n_input:,:] = temp*-1
connections[0] = b.Connection(input_groups['input'], neuron_groups[0], structure= conn_structure, 
                                            state = 'v', delay=False)#, max_delay=delay[connType][1])
connections[0].connect(input_groups['input'], neuron_groups[0], weight_matrix)#, delay=delay[connType])
     
#------------------------------------------------------------------------------ 
# create connections for hidden layers
#------------------------------------------------------------------------------ 
for i in xrange(num_layers-1):
    prev_pop = i
#     print 'create connections between', prev_pop, 'and', i+1
    weight_matrix = np.loadtxt(weight_path + str(i+1) + '.txt').reshape((num_neurons[i], num_neurons[i+1]))
    print np.min(weight_matrix), np.max(weight_matrix)
    src_pop = neuron_groups[prev_pop]
    tgt_pop = neuron_groups[i+1]
#     print weight_matrix.shape
    connections[i+1] = b.Connection(src_pop, tgt_pop, structure= conn_structure, 
                                                state = 'v', delay=False)#, max_delay=delay[connType][1])
    connections[i+1].connect(neuron_groups[prev_pop], neuron_groups[i+1], weight_matrix)#, delay=delay[connType])
     
     
     
#------------------------------------------------------------------------------ 
# load word2vec
#------------------------------------------------------------------------------ 
# print 'load word2vec'
layer0 = np.asarray(np.loadtxt(weight_path + '0.txt'))
layer1 = np.asarray(np.loadtxt(weight_path + '1.txt'))
model = Word2Vec.load_word2vec_format(weight_path + 'word2vec.bin', binary=True, norm_only=False)
# print model.most_similar(positive=['woman', 'prince'], negative=['man'])

# import brian.experimental.realtime_monitor as rltmMon
# real_time_monitor = rltmMon.RealtimeConnectionMonitor(connections['AeAe'], cmap=cmap.get_cmap('gist_ncar'), wmin=0, wmax=wmax_ee, clock = b.Clock(5000*b.ms))

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = 0
while True:
    try:
        word = raw_input("How happy is: ")
        if word==exit:
            break
        raw_vector = np.asarray(model[word])
#         print (raw_vector)
        layer0_output = np.maximum(0,np.dot(raw_vector, layer0))
        layer1_output = np.dot(layer0_output, layer1)
        print 'Float output:', layer1_output
#         print np.round(raw_vector*10)/10.
#         rounded_vector = np.round(raw_vector*10)/10.
#         layer0_output = np.maximum(0,np.dot(rounded_vector, layer0))
#         layer1_output = np.dot(layer0_output, layer1)
#         print 'Float output:', layer1_output
            
        np.random.seed(1000)
        for i in xrange(num_layers):
            neuron_groups[i].v = np.zeros((num_neurons[i]))
                
        if poisson_inputs:
            rates = np.zeros((n_input*2))
            # positive inputs
            rates[:n_input] = np.clip(raw_vector*max_input_rate/single_example_time, 0, 1./timestep)
            # negative inputs
            rates[n_input:] = np.clip(-1*raw_vector*max_input_rate/single_example_time, 0, 1./timestep)
            input_groups['input'].rate = rates
        else:
            spiketimes = []
            num_timesteps = single_example_time/timestep
            for input_neuron in xrange(n_input):
                # positive inputs
#                 print 'raw_vector[input_neuron]*num_timesteps', raw_vector[input_neuron]*num_timesteps
                times = np.linspace(0, num_timesteps-num_layers-1, np.clip(raw_vector[input_neuron]*num_timesteps, 0, 1./timestep))
                spikes = zip([input_neuron]*len(times), times)
#                 print 'spiketimes', spiketimes
#                 print 'spikes', spikes
                if spikes != []:
                    if spiketimes == []:
                        spiketimes = spikes
                    else:
                        spiketimes = np.concatenate((spiketimes, spikes), axis=0)
                # negative inputs
                times = np.linspace(0, num_timesteps-num_layers-1, np.clip(-1*raw_vector[input_neuron]*num_timesteps, 0, 1./timestep))
                spikes = zip([input_neuron+n_input]*len(times), times)
#                 print 'spiketimes', spiketimes
#                 print 'spikes', spikes
                if spikes != []:
                    if spiketimes == []:
                        spiketimes = spikes
                    else:
#                         print spiketimes
                        spiketimes = np.concatenate((spiketimes, spikes), axis=0)
            input_groups['input'].spiketimes = [(y,z) for y,z in spiketimes]
            print input_groups['input'].spiketimes
            
                
        b.run(single_example_time)#, report='text')
                            
        current_spike_count = np.asarray(spike_counters[num_layers-1].count[:]) - previous_spike_count
        #     print current_spike_count,  np.asarray(spike_counters['Ce'].count[:]), previous_spike_count
        previous_spike_count = np.copy(spike_counters[num_layers-1].count[:])
        print "Number of output spikes:", current_spike_count, '\n'
        imshow(img_dict[(int(round(layer1_output)))])
        draw()
            
        #     b.figure()
        #     b.plot(state_monitors[0].times/b.second, state_monitors[0]['v'][0], label = ' v 0')
        #     b.title('membrane voltages of population ')
        #     b.show()
    
    except KeyError:
        print word, ' is not/rarely mentioned in Wikipedia. Please choose something more common...\n'
    



b.ioff()
b.show()



























