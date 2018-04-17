import os
import glob
import pandas as pd

def get_events(f):
  with open(f, 'r') as f:
    events = []
    line_num = 0
    event_line_num = -1
    print("Reading data from {} ...".format(f))
    for line in f:
      if line_num % 100000 == 0:
        print("\tline {}".format(line_num))
      if '<event>' in line:
        event_line_num = 0
        event = []
      elif '</event>' in line:
        events.append(event)
        event_line_num = -1
      elif event_line_num == 1:
        num_particles = float(line[0])
        if num_particles != 6:
          print("Error in file {} on line {}:\n\t{}\n\tExpected 6 particles in event, but found {}.".format(f, line_num, line, num_particles))
      elif event_line_num != -1:
        event.extend(float(s) for s in line.split())
      if event_line_num != -1:
        event_line_num += 1
      line_num += 1
    events = pd.DataFrame(events, columns=pd.MultiIndex.from_product([['particle_1', 'particle_2', 'particle_3', 'particle_4', 'particle_5', 'particle_6'], ['pdgid', 'status', 'parent_1', 'parent_2', 'color_1', 'color_2', 'px', 'py', 'pz', 'E', 'm', 'lifetime', 'spin']]))
    print("Done.")
    return events

def validate_file(f):
  events = get_events(f)
  print("Checking initial particles...")
  print("\tRequiring sum of three-momenta to be zero.")
  assert(((events['particle_1'][['px', 'py', 'pz']] + events['particle_2'][['px', 'py', 'pz']]) == 0.0).all().all())
  print("\tRequiring total energy to be 1500GeV.")
  assert(((events['particle_1']['E'] + events['particle_2']['E']) == 1500.0).all())
  print("Checking Higgs decay...")
  print("\tRequiring invariant mass to be 125GeV.")
  Ehiggs = (events['particle_3']['E'] + events['particle_4']['E'])**2
  phiggs = (events['particle_3'][['px', 'py', 'pz']] + events['particle_4'][['px', 'py', 'pz']])**2
  mhiggs = (Ehiggs - phiggs.sum(axis=1))**0.5
  assert((round(mhiggs, 0) == 125.0).all())
  print("Checking Z decay...")
  print("\tRequiring invariant mass to be 91GeV.")
  EZ = (events['particle_5']['E'] + events['particle_6']['E'])**2
  pZ = (events['particle_5'][['px', 'py', 'pz']] + events['particle_6'][['px', 'py', 'pz']])**2
  mZ = (EZ - pZ.sum(axis=1))**0.5
  assert((round(mZ, 0) == 91.0).all())

def validate(path):
  if not os.path.exists(path):
    raise OSError("{} is not a valid file or directory!".format(path))
  if os.path.isdir(path):
    files = glob.glob(os.path.join(path, "*.lhe"))
    for f in files:
      validate_file(f)
  else:
    validate_file(path)

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Validate a LHE file or set of files.')
  parser.add_argument('file', type=str, help='a LHE file or a directory containing LHE files that should be validated.')
  args = parser.parse_args()
  validate(args.file)
  


if __name__ == '__main__':
  main()
