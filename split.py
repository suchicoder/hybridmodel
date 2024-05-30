def split_text(text):
    # List of keywords to identify minor use cases
    keywords = ['Title', 'Use CaseName', 'Use Case', 'Requirement No.', 'Description', 'Name', 'actors']
    
    # Initialize a dictionary to store minor use cases
    minor_use_cases = {}
    current_use_case = ''
    
    # Split the text based on keywords
    lines = text.split('\n')
    for line in lines:
        # Check if the line contains any keyword
        found_keyword = False
        for keyword in keywords:
            if keyword.lower() in line.lower():
                found_keyword = True
                if current_use_case:
                    minor_use_cases[current_use_case].append(line)
                else:
                    current_use_case = line
                    minor_use_cases[current_use_case] = []
                break
        
        # If no keyword is found, add the line to the current use case
        if not found_keyword and current_use_case:
            minor_use_cases[current_use_case].append(line)
    
    return minor_use_cases


minor_use_cases = split_text(processed_text)
for use_case, details in minor_use_cases.items():
    print(f"{use_case}:")
    print("\n".join(details))
    print()
    '
