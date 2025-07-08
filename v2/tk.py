import pygame
import requests
import re

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Server Info")
clock = pygame.time.Clock()

regular_font = pygame.font.SysFont('arial', 16)
bold_font = pygame.font.SysFont('arial', 16, bold=True)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def parse_description(text):
    segments = []
    stack = [{}]
    pos = 0

    patterns = [
        (r'\[color=(\d+),(\d+),(\d+)\]', lambda m: {'color': (int(m.group(1)), int(m.group(2)), int(m.group(3)))}),
        (r'\[/color\]', lambda m: {'end_color': True}),
        (r'\[bold\]', lambda m: {'bold': True}),
        (r'\[/bold\]', lambda m: {'end_bold': True})
    ]

    while pos < len(text):
        next_match = None
        next_pattern = None
        min_start = len(text)

        for pattern, action in patterns:
            match = re.search(pattern, text[pos:])
            if match and pos + match.start() < min_start:
                min_start = pos + match.start()
                next_match = match
                next_pattern = action

        if next_match:
            start = min_start
            if pos < start:
                segments.append({
                    'text': text[pos:start],
                    'color': stack[-1].get('color', WHITE),
                    'bold': stack[-1].get('bold', False)
                })
            tag_data = next_pattern(next_match)
            if 'end_color' in tag_data or 'end_bold' in tag_data:
                stack.pop()
            else:
                stack.append({**stack[-1], **tag_data})
            pos = start + len(next_match.group(0))
        else:
            segments.append({
                'text': text[pos:],
                'color': stack[-1].get('color', WHITE),
                'bold': stack[-1].get('bold', False)
            })
            break

    return [s for s in segments if s['text']]

def wrap_text(segments, max_chars=25, max_lines=4):
    lines = []
    current_line = []
    current_length = 0

    for segment in segments:
        words = segment['text'].split()
        for word in words:
            word_len = len(word)
            space_len = 1 if current_length > 0 else 0
            if current_length + word_len + space_len <= max_chars:
                current_line.append({
                    'text': word,
                    'color': segment['color'],
                    'bold': segment['bold']
                })
                current_length += word_len + space_len
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = []
                    current_length = 0
                if word_len > max_chars:
                    word = word[:max_chars]
                    word_len = len(word)
                current_line.append({
                    'text': word,
                    'color': segment['color'],
                    'bold': segment['bold']
                })
                current_length = word_len
                if len(lines) == max_lines - 1 and current_length >= max_chars:
                    lines.append(current_line)
                    return lines

        if len(lines) == max_lines:
            break

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    return lines

def render_lines(lines, x, y, line_spacing=20):
    for i, line in enumerate(lines):
        current_x = x
        for segment in line:
            font = bold_font if segment['bold'] else regular_font
            text_surface = font.render(segment['text'], True, segment['color'])
            screen.blit(text_surface, (current_x, y + i * line_spacing))
            current_x += text_surface.get_width() + regular_font.render(' ', True, WHITE).get_width()

def main():
    try:
        response = requests.get('https://geomit25.pythonanywhere.com/server_info')
        response.raise_for_status()
        data = response.json()
        description = data.get('description', 'No description available')
        version = data.get('version', 'Unknown')
    except requests.RequestException:
        description = 'Failed to fetch server info'
        version = 'Unknown'

    segments = parse_description(description)
    lines = wrap_text(segments)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        render_lines(lines, 20, 50)
        version_surface = regular_font.render(f"Version: {version}", True, WHITE)
        screen.blit(version_surface, (20, 150))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
