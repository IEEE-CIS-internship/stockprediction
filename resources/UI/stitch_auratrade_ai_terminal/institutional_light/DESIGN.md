---
name: Institutional Light
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#45474c'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e72'
  primary: '#040c1c'
  on-primary: '#ffffff'
  primary-container: '#1a2233'
  on-primary-container: '#81899e'
  inverse-primary: '#bec6dd'
  secondary: '#006780'
  on-secondary: '#ffffff'
  secondary-container: '#76dcff'
  on-secondary-container: '#006077'
  tertiary: '#1a0600'
  on-tertiary: '#ffffff'
  tertiary-container: '#3d1700'
  on-tertiary-container: '#d66c26'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fa'
  primary-fixed-dim: '#bec6dd'
  on-primary-fixed: '#131b2c'
  on-primary-fixed-variant: '#3f4759'
  secondary-fixed: '#b7eaff'
  secondary-fixed-dim: '#6cd3f7'
  on-secondary-fixed: '#001f28'
  on-secondary-fixed-variant: '#004e61'
  tertiary-fixed: '#ffdbca'
  tertiary-fixed-dim: '#ffb68e'
  on-tertiary-fixed: '#331200'
  on-tertiary-fixed-variant: '#763300'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  3xl: 64px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
---

## Brand & Style
The design system is engineered for high-stakes financial environments and professional presentations. It shifts the aesthetic from a dense, screen-focused dark mode to an expansive, high-visibility "Institutional Light" framework. The personality is authoritative, transparent, and precise, optimized for clarity on projectors and high-resolution displays alike.

The style leverages **Corporate Modernism** with a focus on tonal layering. It utilizes a sophisticated off-white foundation to reduce eye strain while using pure white surfaces to denote interactive or prioritized data containers. The emotional response is one of calm confidence—moving away from the "gamer" aesthetic of traditional crypto-tools toward a refined, bank-grade analytical platform.

## Colors
The palette is anchored by a deep navy-charcoal for maximum text legibility. The background employs a soft cool-grey to define the workspace boundaries without the harshness of pure white.

- **Primary Surfaces**: #F5F7FB provides a stable environment for long-term data monitoring.
- **Elevation Surfaces**: #FFFFFF is used exclusively for interactive cards and modules to create a clear "layering" effect.
- **Semantic Signals**: Green and Red values are weighted for high contrast against light backgrounds. Bullish signals use a deeper emerald to maintain presence, while Bearish signals use a saturated coral to ensure urgency.
- **Accents**: Muted Teal/Cyan provides a technical feel for focus states and primary actions, while Deep Gold is reserved for AI-driven insights and "Alpha" highlights.

## Typography
This design system utilizes **Geist** for its primary and body roles, benefitting from its technical precision and geometric clarity. For data-heavy contexts—such as price tickers, transaction hashes, and coordinates—**JetBrains Mono** is introduced to provide a distinctive, monospaced "Developer-Core" aesthetic that signals accuracy.

Headline weights are kept medium-to-bold to anchor sections, while body text remains regular for maximum breathability. Letter spacing is slightly tightened on large headings to maintain a compact, "institutional" feel, whereas labels use expanded tracking for better legibility at small scales.

## Layout & Spacing
The layout follows a strict **Fluid Grid** model based on an 8px rhythmic scale. For desktop views, a 12-column grid with a 24px gutter is used to organize complex dashboards. 

- **Breathing Room**: Generous internal padding (24px) within cards prevents data density fatigue.
- **Reflow**: On mobile, the grid collapses to a 4-column system. Cards should stretch full-width, utilizing 16px side margins to maximize space for technical charts.
- **Alignment**: Information should be grouped into logical modules with "xl" (32px) spacing between disparate functional areas.

## Elevation & Depth
Depth is communicated through **Ambient Shadows** and **Tonal Layering**. Unlike the dark variant which relies on light-borders, the light variant uses soft shadows to create a physical sense of "float."

- **Level 0 (Background)**: #F5F7FB. The lowest plane.
- **Level 1 (Cards/Modules)**: #FFFFFF with a very soft, large spread shadow: `0px 10px 30px rgba(0,0,0,0.05)`. This level is where most user interaction occurs.
- **Level 2 (Dropdowns/Modals)**: #FFFFFF with a more pronounced shadow: `0px 20px 40px rgba(0,0,0,0.08)`. These elements should appear clearly above the standard card plane.
- **Borders**: Subtle 1px borders (#E2E8F0) are used on all Level 1 and Level 2 elements to define edges when the light source is direct or screen brightness is high.

## Shapes
The design system utilizes a **Rounded** shape language to soften the industrial nature of financial data.

- **Standard Containers**: All cards and primary UI blocks use a 12px (0.75rem) corner radius.
- **Interactive Elements**: Buttons and input fields follow the 8px (0.5rem) radius to maintain a tighter, more functional look.
- **Status Indicators**: Small badges and chips use a "Pill" radius (100px) to distinguish them from structural elements.

## Components
- **Buttons**: Primary buttons use the Accent Teal (#0891B2) with white text. Secondary buttons use a white fill with a subtle slate border (#E2E8F0) and navy-charcoal text.
- **Cards**: Always white background, 12px rounded corners, and the signature soft ambient shadow. Use 24px padding for all standard content.
- **Input Fields**: Background should be #FFFFFF with a 1px border (#E2E8F0). On focus, the border transitions to Accent Teal (#0891B2) with a subtle glow.
- **Chips/Badges**: For bullish/bearish signals, use a 10% opacity background of the semantic color with 100% opacity text of the same color (e.g., Emerald Green text on a faint green background).
- **Data Tables**: Use #F5F7FB for the header background and 1px horizontal dividers (#E2E8F0). Avoid vertical dividers to keep the look modern and clean.
- **AI Insight Module**: Highlighted using the Deep Gold (#B45309) as a left-hand accent border (4px width) to draw immediate attention.