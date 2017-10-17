

def ReadLocations(filename):
	with open(filename, "r") as ins:
		array = []
		for line in ins:
			array.append(line.rstrip('\n'))
		return array
